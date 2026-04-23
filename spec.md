# QRODE v0.2 Concrete Module and Interface Spec

This document defines implementation-ready interfaces for the Rust-first architecture.

## Locked Decisions

1. ECC level default is `M`.
2. Encoding mode is `Byte` only in v0.2.
3. Payload is always full capacity for selected `(version, ecc=M, mode=Byte)`.
4. Payload format in v0.2 is arbitrary bytes.
5. Default optimizer is simulated annealing with random restarts.
6. Score objective optimizes mutable-region quality and reports full-grid quality separately.

## Global Terms and Invariants

1. Grid size for version `v` is `n = 21 + 4 * (v - 1)`.
2. `target[y][x]` and QR module matrices use `bool` where `true = dark`, `false = light`.
3. Immutable modules are those not controlled by payload bits for chosen `(version, ecc, mask)`.
4. Payload layout:
   - `payload = prefix || suffix`
   - `payload.len() == capacity(version, M, Byte)`
   - `prefix.len() <= payload.len()`
5. If `prefix` does not fit, return capacity error or require higher version.

## Crate Module Map

1. `src/qr_core/mod.rs`
2. `src/target_adapter.rs`
3. `src/scoring.rs`
4. `src/optimizer/mod.rs`
5. `src/validator.rs`
6. `src/cli.rs`
7. `src/main.rs`

## Common Types

Define these shared types in `qr_core` or a future `types` module and re-export where needed.

```rust
pub type Module = bool;
pub type Matrix = Vec<Vec<Module>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EccLevel {
    L,
    M,
    Q,
    H,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataMode {
    Byte,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaskPattern {
    M0,
    M1,
    M2,
    M3,
    M4,
    M5,
    M6,
    M7,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QrSpec {
    pub version: u8,
    pub ecc: EccLevel,
    pub mode: DataMode,
}

#[derive(Clone, Debug)]
pub struct EncodedQr {
    pub spec: QrSpec,
    pub mask: MaskPattern,
    pub modules: Matrix,
}

#[derive(thiserror::Error, Debug)]
pub enum QrodeError {
    #[error("unsupported QR version: {0}")]
    UnsupportedVersion(u8),
    #[error("payload length mismatch: got {got}, expected {expected}")]
    PayloadLengthMismatch { got: usize, expected: usize },
    #[error("prefix length {prefix_len} exceeds capacity {capacity}")]
    PrefixTooLong { prefix_len: usize, capacity: usize },
    #[error("image decode/parse error: {0}")]
    Image(String),
    #[error("decode validation failed: {0}")]
    Validation(String),
    #[error("internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, QrodeError>;
```

## Module Interfaces

### 1) `qr_core`

Responsibilities:

1. Capacity lookup for `(version, ecc=M, mode=Byte)`.
2. Encode payload bytes to module matrix.
3. Provide immutable module mask for scoring.

Public API:

```rust
pub fn capacity_bytes(spec: QrSpec) -> Result<usize>;

pub fn encode_with_mask(spec: QrSpec, payload: &[u8], mask: MaskPattern) -> Result<EncodedQr>;

pub fn encode_auto_mask(spec: QrSpec, payload: &[u8]) -> Result<EncodedQr>;

/// true = immutable module, false = mutable module
pub fn immutable_mask(spec: QrSpec, mask: MaskPattern) -> Result<Matrix>;

pub fn module_size(version: u8) -> Result<usize>;
```

Contract notes:

1. `capacity_bytes` returns exact data byte capacity for byte mode.
2. `encode_*` must reject any payload length not equal to capacity.
3. `immutable_mask` must align exactly with encoded matrix dimensions.

### 2) `target_adapter`

Responsibilities:

1. Convert input image to QR-sized target matrix.
2. Deterministic geometry policy (`contain`/`cover`).
3. Binarize grayscale to module values.

Public API:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FitMode {
    Contain,
    Cover,
}

#[derive(Clone, Copy, Debug)]
pub struct TargetAdapterConfig {
    pub fit_mode: FitMode,
    pub threshold: u8, // 0..=255
    pub invert: bool,
}

pub fn load_and_adapt_target(
    image_path: &std::path::Path,
    version: u8,
    config: TargetAdapterConfig,
) -> Result<Matrix>;
```

Contract notes:

1. Output is always `n x n` for given version.
2. Same input and config yields identical output.

### 3) `scoring`

Responsibilities:

1. Compute mutable-region objective for optimization.
2. Compute full-grid reporting metric.
3. Optionally apply immutable down-weighting.

Public API:

```rust
#[derive(Clone, Copy, Debug)]
pub struct ScoreWeights {
    pub mutable_weight: f32,
    pub immutable_weight: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ScoreBreakdown {
    pub objective_score: f32,      // optimizer target
    pub mutable_match_pct: f32,    // 0..100
    pub full_match_pct: f32,       // 0..100
    pub mutable_dark_error: f32,
    pub mutable_light_error: f32,
}

pub fn score_modules(
    candidate: &Matrix,
    target: &Matrix,
    immutable_mask: &Matrix,
    weights: ScoreWeights,
) -> Result<ScoreBreakdown>;
```

Contract notes:

1. `objective_score` should be monotonic with optimization quality.
2. v0.2 default: optimize `mutable_match_pct` primarily.

### 4) `optimizer`

Responsibilities:

1. Search mutable suffix bytes with fixed prefix and fixed total length.
2. Support annealing acceptance and restart strategy.
3. Return best payload and metadata.

Public API:

```rust
#[derive(Clone, Debug)]
pub struct OptimizeConfig {
    pub spec: QrSpec,
    pub fixed_mask: Option<MaskPattern>,
    pub search_masks: bool, // if true, evaluate masks 0..7
    pub seed: u64,
    pub max_iters: u64,
    pub restarts: u32,
    pub init_temp: f32,
    pub final_temp: f32,
    pub byte_mutation_rate: f32,
    pub bit_mutation_rate: f32,
    pub block_mutation_prob: f32,
    pub guided_mutation_prob: f32,
}

#[derive(Clone, Debug)]
pub struct OptimizeInput {
    pub prefix: Vec<u8>,
    pub target: Matrix,
    pub weights: ScoreWeights,
}

#[derive(Clone, Debug)]
pub struct OptimizeResult {
    pub best_payload: Vec<u8>,
    pub best_encoded: EncodedQr,
    pub best_score: ScoreBreakdown,
    pub iterations: u64,
    pub restarts_done: u32,
    pub elapsed_ms: u128,
    pub seed: u64,
}

pub fn optimize(input: OptimizeInput, config: OptimizeConfig) -> Result<OptimizeResult>;
```

Contract notes:

1. `best_payload.len()` must equal capacity.
2. `best_payload[..prefix.len()] == prefix`.
3. Must be reproducible with same seed, config, inputs, and deterministic dependencies.

Mutation operators in v0.2:

1. Single bit flip in mutable suffix.
2. Single byte random replace.
3. Small contiguous block byte mutation.
4. Guided mutation hook (initially simple frequency map, can improve later).

Acceptance in v0.2:

1. Greedy if score improves.
2. Annealed acceptance for worse proposals with temperature schedule.

### 5) `validator`

Responsibilities:

1. Validate payload policy constraints.
2. Optional external decoder verification in benchmark mode.

Public API:

```rust
#[derive(Clone, Debug)]
pub struct ValidationReport {
    pub prefix_ok: bool,
    pub length_ok: bool,
    pub decode_ok: bool,
    pub decoded_payload: Option<Vec<u8>>,
    pub message: String,
}

pub fn validate_payload_policy(
    payload: &[u8],
    prefix: &[u8],
    expected_len: usize,
) -> Result<ValidationReport>;

pub fn validate_with_decoder(
    png_path: &std::path::Path,
    expected_payload: &[u8],
) -> Result<ValidationReport>;
```

Contract notes:

1. Decoder validation is optional for fast runs and required for benchmark runs.

### 6) `cli`

Responsibilities:

1. Parse runtime configuration.
2. Build target matrix, run optimizer, write artifacts, print report.

Runtime presets:

1. `quick`: budget approx. 5-15 seconds.
2. `quality`: budget approx. 1-5 minutes.

Public API:

```rust
#[derive(clap::Parser, Debug)]
pub struct CliArgs {
    #[arg(long)]
    pub target: std::path::PathBuf,
    #[arg(long)]
    pub out_png: std::path::PathBuf,
    #[arg(long)]
    pub out_json: std::path::PathBuf,
    #[arg(long)]
    pub version: u8,
    #[arg(long, default_value = "M")]
    pub ecc: String,
    #[arg(long, default_value = "")]
    pub prefix_hex: String,
    #[arg(long, default_value = "quick")]
    pub mode: String,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub max_iters: Option<u64>,
    #[arg(long)]
    pub restarts: Option<u32>,
    #[arg(long)]
    pub mask: Option<u8>,
    #[arg(long, default_value_t = false)]
    pub search_masks: bool,
    #[arg(long, default_value_t = false)]
    pub benchmark_decode: bool,
}

pub fn run(args: CliArgs) -> Result<()>;
```

CLI output report fields (JSON):

1. `version`, `ecc`, `mask`, `seed`
2. `capacity_bytes`, `prefix_len`, `suffix_len`
3. `iterations`, `restarts`, `elapsed_ms`
4. `mutable_match_pct`, `full_match_pct`, `objective_score`
5. `decode_ok`, `decoded_payload_hex` (if enabled)

### 7) `main`

Responsibilities:

1. Parse args and call `cli::run`.
2. Exit with non-zero code on error.

## End-to-End Flow

1. Parse CLI args.
2. Construct `QrSpec { version, ecc: M, mode: Byte }`.
3. Resolve capacity and validate prefix length.
4. Build full-length candidate payload with mutable suffix.
5. Adapt target image to `n x n` matrix.
6. Run optimizer and collect best result.
7. Render final QR PNG and JSON report.
8. If benchmark mode enabled, run decoder validation.

## Benchmark Protocol (v0.2)

Purpose:

1. Compare optimizer variants and runtime presets reproducibly.

Fixed benchmark inputs:

1. A small suite of target images (at least 5, mixed structure complexity).
2. Versions tested (for example: 5, 8, 12).
3. Prefix lengths (for example: 0, 8, 24 bytes).
4. Seeds (at least 5 deterministic seeds per case).

Per-run recorded metrics:

1. Best mutable match percent.
2. Best full-grid match percent.
3. Iterations to best.
4. Wall-clock runtime.
5. Decode success rate (when decoder enabled).

Comparison rules:

1. Report mean, median, and p90 for each metric.
2. Use same seeds and image set across algorithm variants.
3. Treat decode failure as hard failure for benchmark leaderboards.

## Future Extension Hooks

1. Constrained printable/URL mode:
   - Add charset-constrained mutable suffix operators.
   - Preserve fixed required prefix.
2. Multi-version coarse-to-fine:
   - Seed higher version from lower-version optimized payload.
3. Optional Python research harness:
   - Bind Rust optimizer for plotting and A/B experimentation without moving core loop out of Rust.

## Implementation Order

1. `qr_core` capacity + encode + immutable mask.
2. `target_adapter` deterministic resize + threshold.
3. `scoring` objective + diagnostics.
4. `optimizer` annealing + restarts.
5. `cli` wiring and report writing.
6. `validator` benchmark decode integration.
