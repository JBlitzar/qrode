use image::{ImageBuffer, Luma};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::optimizer::{OptimizeConfig, OptimizeInput, optimize};
use crate::qr_core::{
    DataMode, EccLevel, MaskPattern, QrSpec, QrodeError, Result, capacity_bytes, encode_with_mask,
};
use crate::scoring::ScoreWeights;
use crate::target_adapter::{
    FitMode, TargetAdapterConfig, load_and_adapt_target, synthetic_circle_target,
};
use crate::validator::{validate_payload_policy, validate_with_decoder};

const URL_MODE_PREFIX: &str = "https://github.com/JBlitzar/qrode#";

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum PayloadMode {
    Binary,
    Url,
}

#[derive(clap::Parser, Debug)]
pub struct CliArgs {
    #[arg(long)]
    pub target: Option<PathBuf>,
    #[arg(long)]
    pub frames_dir: Option<PathBuf>,
    #[arg(long, default_value = "circle")]
    pub target_kind: String,
    #[arg(long, default_value = "out.png")]
    pub out_png: PathBuf,
    #[arg(long, default_value = "out.json")]
    pub out_json: PathBuf,
    #[arg(long, default_value_t = 20)]
    pub version: u8,
    #[arg(long, default_value = "H")]
    pub ecc: String,
    #[arg(long, default_value = "")]
    pub prefix_hex: String,
    #[arg(long, value_enum, default_value = "binary")]
    pub payload_mode: PayloadMode,
    #[arg(long, default_value = "quick")]
    pub mode: String,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub max_iters: Option<u64>,
    #[arg(long, default_value_t = 0)]
    pub progress_every_iters: u64,
    #[arg(long, default_value_t = 10)]
    pub progress_every_secs: u64,
    #[arg(long)]
    pub time_budget_secs: Option<u64>,
    #[arg(long)]
    pub restarts: Option<u32>,
    #[arg(long)]
    pub mask: Option<u8>,
    #[arg(long, default_value_t = true)]
    pub search_masks: bool,
    #[arg(long = "no-image-space-perturb", action = clap::ArgAction::SetFalse, default_value_t = true)]
    pub enable_image_space_perturb: bool,
    #[arg(long = "no-ssim", action = clap::ArgAction::SetFalse, default_value_t = true)]
    pub use_ssim: bool,
    #[arg(long, default_value_t = false)]
    pub benchmark_decode: bool,
    #[arg(long, default_value_t = 128)]
    pub threshold: u8,
    #[arg(long, default_value_t = 20.0)]
    pub frame_seed_diff_pct: f32,
    #[arg(long, default_value_t = 500_000)]
    pub seeded_frame_iters: u64,
    #[arg(long, default_value_t = 2_000)]
    pub min_seeded_iters: u64,
    #[arg(long, default_value_t = 0.10)]
    pub objective_target: f32,
    #[arg(long, default_value_t = false)]
    pub write_json: bool,
    #[arg(long)]
    pub frame_limit: Option<usize>,
}

#[derive(Serialize)]
struct RunReport {
    version: u8,
    ecc: String,
    mask: u8,
    seed: u64,
    capacity_bytes: usize,
    prefix_len: usize,
    suffix_len: usize,
    iterations: u64,
    restarts: u32,
    elapsed_ms: u128,
    ssim: f32,
    mutable_match_pct: f32,
    full_match_pct: f32,
    objective_score: f32,
    decode_ok: Option<bool>,
    decoded_payload_hex: Option<String>,
}

#[derive(Clone)]
struct FrameJob {
    index: usize,
    path: PathBuf,
    stem: String,
    target: Vec<Vec<bool>>,
}

#[derive(Clone)]
struct SeedCacheEntry {
    frame_index: usize,
    target: Vec<Vec<bool>>,
    payload: Vec<u8>,
}

fn collect_frame_paths(frames_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut frames = Vec::new();
    let entries = std::fs::read_dir(frames_dir)
        .map_err(|e| QrodeError::Internal(format!("failed to read frames dir: {e}")))?;

    for entry in entries {
        let entry =
            entry.map_err(|e| QrodeError::Internal(format!("failed to read dir entry: {e}")))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !name.starts_with("output_") {
            continue;
        }
        let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
            continue;
        };
        if !matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png") {
            continue;
        }
        frames.push(path);
    }

    frames.sort();
    if frames.is_empty() {
        return Err(QrodeError::Internal(format!(
            "no frames found in {} matching output_*.jpg/jpeg/png",
            frames_dir.display()
        )));
    }
    Ok(frames)
}

fn changed_pixel_ratio(current: &[Vec<bool>], previous: &[Vec<bool>]) -> Result<f32> {
    if current.len() != previous.len() || current.is_empty() {
        return Err(QrodeError::Internal(
            "cannot compare frame targets with mismatched dimensions".into(),
        ));
    }
    if current[0].len() != previous[0].len() {
        return Err(QrodeError::Internal(
            "cannot compare frame targets with mismatched dimensions".into(),
        ));
    }

    let mut changed = 0usize;
    let mut total = 0usize;
    for (row_cur, row_prev) in current.iter().zip(previous.iter()) {
        for (&px_cur, &px_prev) in row_cur.iter().zip(row_prev.iter()) {
            total += 1;
            if px_cur ^ px_prev {
                changed += 1;
            }
        }
    }

    if total == 0 {
        return Ok(1.0);
    }
    Ok(changed as f32 / total as f32)
}

fn frame_artifact_path(base: &Path, frame_stem: &str) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new("."));
    let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let ext = base.extension().and_then(|s| s.to_str()).unwrap_or("png");
    parent.join(format!("{stem}_{frame_stem}.{ext}"))
}

fn hash_target(target: &[Vec<bool>]) -> u64 {
    let mut hasher = DefaultHasher::new();
    target.len().hash(&mut hasher);
    for row in target {
        row.len().hash(&mut hasher);
        for &cell in row {
            cell.hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn perturb_seed_payload(
    seed_payload: &[u8],
    prefix_len: usize,
    allowed_mutable_bytes: Option<&[u8]>,
    noise_seed: u64,
    diff_ratio: f32,
) -> Vec<u8> {
    let mut out = seed_payload.to_vec();
    if prefix_len >= out.len() {
        return out;
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(noise_seed);
    let mutable_len = out.len() - prefix_len;
    let mut edits = if diff_ratio <= 0.02 {
        2usize
    } else if diff_ratio <= 0.08 {
        1usize
    } else {
        0usize
    };
    if mutable_len < edits {
        edits = mutable_len;
    }

    for _ in 0..edits {
        let idx = rng.gen_range(prefix_len..out.len());
        if let Some(allowed) = allowed_mutable_bytes {
            if !allowed.is_empty() {
                out[idx] = allowed[rng.gen_range(0..allowed.len())];
            }
        } else {
            let bit = rng.gen_range(0..8);
            out[idx] ^= 1u8 << bit;
        }
    }

    out
}

fn nearest_seed_from_cache(
    target: &[Vec<bool>],
    cache: &[SeedCacheEntry],
) -> Result<Option<(usize, f32, Vec<u8>)>> {
    let mut best: Option<(usize, f32, Vec<u8>)> = None;
    for entry in cache {
        let diff_ratio = changed_pixel_ratio(target, &entry.target)?;
        match &best {
            Some((_, best_diff, _)) if diff_ratio >= *best_diff => {}
            _ => {
                best = Some((entry.frame_index, diff_ratio, entry.payload.clone()));
            }
        }
    }
    Ok(best)
}

fn configure_frame_progress(total: usize) -> ProgressBar {
    let pb = ProgressBar::new(total as u64);
    let style = ProgressStyle::with_template(
        "{bar:40.cyan/blue} {pos:>5}/{len:5} frames | {percent:>3}% | ETA {eta_precise}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-");
    pb.set_style(style);
    pb
}

fn parse_prefix_hex(s: &str) -> Result<Vec<u8>> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    if !s.len().is_multiple_of(2) {
        return Err(QrodeError::Internal(
            "prefix_hex must have even length".into(),
        ));
    }

    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = (bytes[i] as char)
            .to_digit(16)
            .ok_or_else(|| QrodeError::Internal("invalid prefix_hex".into()))?;
        let lo = (bytes[i + 1] as char)
            .to_digit(16)
            .ok_or_else(|| QrodeError::Internal("invalid prefix_hex".into()))?;
        out.push(((hi << 4) | lo) as u8);
    }
    Ok(out)
}

fn urlsafe_charset() -> Vec<u8> {
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~%!$&'()*+,;=:@/".to_vec()
}

fn write_qr_png(modules: &[Vec<bool>], out_png: &PathBuf) -> Result<()> {
    let module_px = 8u32;
    let quiet_zone = 4u32;
    let n = modules.len() as u32;
    let img_size = (n + 2 * quiet_zone) * module_px;

    let mut img: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_pixel(img_size, img_size, Luma([255]));
    for (y, row) in modules.iter().enumerate() {
        for (x, &dark) in row.iter().enumerate() {
            let color = if dark { 0u8 } else { 255u8 };
            let px0 = (x as u32 + quiet_zone) * module_px;
            let py0 = (y as u32 + quiet_zone) * module_px;
            for py in py0..py0 + module_px {
                for px in px0..px0 + module_px {
                    img.put_pixel(px, py, Luma([color]));
                }
            }
        }
    }
    img.save(out_png)
        .map_err(|e| QrodeError::Internal(format!("failed to save png: {e}")))
}

fn write_bitmap_png(modules: &[Vec<bool>], out_png: &PathBuf) -> Result<()> {
    let module_px = 8u32;
    let n = modules.len() as u32;
    let img_size = n * module_px;

    let mut img: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_pixel(img_size, img_size, Luma([255]));
    for (y, row) in modules.iter().enumerate() {
        for (x, &dark) in row.iter().enumerate() {
            let color = if dark { 0u8 } else { 255u8 };
            let px0 = x as u32 * module_px;
            let py0 = y as u32 * module_px;
            for py in py0..py0 + module_px {
                for px in px0..px0 + module_px {
                    img.put_pixel(px, py, Luma([color]));
                }
            }
        }
    }
    img.save(out_png)
        .map_err(|e| QrodeError::Internal(format!("failed to save bitmap png: {e}")))
}

fn derived_target_out_path(out_png: &PathBuf) -> PathBuf {
    let parent = out_png
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let stem = out_png
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("out");
    parent.join(format!("{stem}.target.png"))
}

pub fn run(args: CliArgs) -> Result<()> {
    let ecc = match args.ecc.as_str() {
        "H" | "h" => EccLevel::H,
        "L" | "l" => EccLevel::L,
        "M" | "m" => EccLevel::M,
        "Q" | "q" => EccLevel::Q,
        _ => {
            return Err(QrodeError::Internal(
                "invalid ECC level. Expected H, L, M, or Q".into(),
            ));
        }
    };

    let spec = QrSpec {
        version: args.version,
        ecc,
        mode: DataMode::Byte,
    };
    let capacity = capacity_bytes(spec)?;

    let (prefix, allowed_mutable_bytes) = match args.payload_mode {
        PayloadMode::Binary => (parse_prefix_hex(&args.prefix_hex)?, None),
        PayloadMode::Url => {
            if !args.prefix_hex.is_empty() {
                return Err(QrodeError::Internal(
                    "prefix_hex cannot be used in url payload mode".into(),
                ));
            }
            (URL_MODE_PREFIX.as_bytes().to_vec(), Some(urlsafe_charset()))
        }
    };
    let _policy = validate_payload_policy(&vec![0; capacity], &prefix, capacity)?;

    let is_circle_target = args.target_kind.eq_ignore_ascii_case("circle");
    let is_frames_target = args.target_kind.eq_ignore_ascii_case("frames");

    let (default_iters, default_restarts, default_time_budget_secs) = match args.mode.as_str() {
        "quick" => (
            5_000_000,
            (rayon::current_num_threads() as u32).clamp(2, 8),
            15u64,
        ),
        "quality" => (
            50_000_000,
            ((rayon::current_num_threads() - 1) as u32).clamp(2, 16),
            55u64,
        ),
        _ => return Err(QrodeError::Internal("mode must be quick or quality".into())),
    };
    let seed = args.seed.unwrap_or(1);
    let time_budget_ms = args
        .time_budget_secs
        .unwrap_or(default_time_budget_secs)
        .saturating_mul(1000) as u128;

    let fixed_mask = args
        .mask
        .map(|m| {
            MaskPattern::from_u8(m).ok_or_else(|| QrodeError::Internal("mask must be 0..7".into()))
        })
        .transpose()?;

    if is_frames_target {
        let frames_dir = args
            .frames_dir
            .clone()
            .or_else(|| args.target.clone())
            .unwrap_or_else(|| PathBuf::from("res/frames"));
        let mut frames = collect_frame_paths(&frames_dir)?;
        if let Some(limit) = args.frame_limit {
            frames.truncate(limit);
        }
        if frames.is_empty() {
            return Err(QrodeError::Internal(
                "frame_limit resulted in zero frames to process".into(),
            ));
        }
        let frame_threshold = (args.frame_seed_diff_pct / 100.0).clamp(0.0, 1.0);

        let mut frame_jobs = Vec::with_capacity(frames.len());
        for (idx, frame_path) in frames.iter().enumerate() {
            let target = load_and_adapt_target(
                frame_path,
                args.version,
                TargetAdapterConfig {
                    fit_mode: FitMode::Contain,
                    threshold: args.threshold,
                    invert: false,
                },
            )?;
            let frame_stem = frame_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("frame")
                .to_string();
            frame_jobs.push(FrameJob {
                index: idx,
                path: frame_path.clone(),
                stem: frame_stem,
                target: target.clone(),
            });
        }

        let pb = configure_frame_progress(frame_jobs.len());
        let run_frame = |job: &FrameJob,
                         initial_payload: Option<Vec<u8>>,
                         seed_source: Option<usize>,
                         seed_diff_ratio: Option<f32>|
         -> Result<Vec<u8>> {
            let seeded_mode = initial_payload.is_some();
            let frame_seed = seed.wrapping_add(job.index as u64 * 1_000_003);
            let mut frame_max_iters = if seeded_mode {
                args.seeded_frame_iters
            } else {
                args.max_iters.unwrap_or(default_iters)
            };
            if seeded_mode {
                frame_max_iters = frame_max_iters.max(args.min_seeded_iters.max(1));
            }
            let frame_restarts = if seeded_mode {
                1
            } else {
                args.restarts.unwrap_or(default_restarts)
            };
            let diff_pct = seed_diff_ratio.unwrap_or(1.0) * 100.0;

            if let Some(source) = seed_source {
                pb.println(format!(
                    "frame={} seeded=true seed_from={} diff={:.3}% iters={} restarts={}",
                    job.path.display(),
                    source + 1,
                    diff_pct,
                    frame_max_iters,
                    frame_restarts
                ));
            } else {
                pb.println(format!(
                    "frame={} seeded=false diff={:.3}% iters={} restarts={}",
                    job.path.display(),
                    diff_pct,
                    frame_max_iters,
                    frame_restarts
                ));
            }

            let frame_png = frame_artifact_path(&args.out_png, &job.stem);
            let frame_json = frame_artifact_path(&args.out_json, &job.stem);
            let input = OptimizeInput {
                prefix: prefix.clone(),
                target: job.target.clone(),
                weights: ScoreWeights {
                    use_ssim: args.use_ssim,
                    ..ScoreWeights::default()
                },
            };

            let mut initial_payload_for_opt = initial_payload;
            if let Some(seed_payload) = initial_payload_for_opt.clone() {
                let seed_diff = seed_diff_ratio.unwrap_or(1.0);
                let precheck_cfg = OptimizeConfig {
                    spec,
                    fixed_mask,
                    search_masks: args.search_masks,
                    initial_payload: Some(seed_payload.clone()),
                    allowed_mutable_bytes: allowed_mutable_bytes.clone(),
                    enable_image_space_perturb: false,
                    coordinate_refine_sweeps: 0,
                    bitwise_seed_init: false,
                    progress_every_iters: None,
                    progress_every_secs: None,
                    progress_image_path: None,
                    seed: frame_seed,
                    max_iters: 0,
                    max_time_ms: None,
                    restarts: 1,
                    single_thread: true,
                    ..OptimizeConfig::default()
                };

                let seed_eval = optimize(input.clone(), precheck_cfg)?;
                if seed_eval.best_score.objective_score >= args.objective_target {
                    let jittered = perturb_seed_payload(
                        &seed_payload,
                        prefix.len(),
                        allowed_mutable_bytes.as_deref(),
                        frame_seed ^ 0x9E37_79B9_7F4A_7C15u64,
                        seed_diff,
                    );
                    if jittered != seed_payload {
                        let jitter_cfg = OptimizeConfig {
                            spec,
                            fixed_mask,
                            search_masks: args.search_masks,
                            initial_payload: Some(jittered.clone()),
                            allowed_mutable_bytes: allowed_mutable_bytes.clone(),
                            enable_image_space_perturb: false,
                            coordinate_refine_sweeps: 0,
                            bitwise_seed_init: false,
                            progress_every_iters: None,
                            progress_every_secs: None,
                            progress_image_path: None,
                            seed: frame_seed ^ 0xA5A5_A5A5_A5A5_A5A5u64,
                            max_iters: 0,
                            max_time_ms: None,
                            restarts: 1,
                            single_thread: true,
                            ..OptimizeConfig::default()
                        };
                        let jitter_eval = optimize(input.clone(), jitter_cfg)?;
                        if jitter_eval.best_score.objective_score >= args.objective_target {
                            initial_payload_for_opt = Some(jittered);
                        }
                    }
                }
            }

            let cfg = OptimizeConfig {
                spec,
                fixed_mask,
                search_masks: args.search_masks,
                initial_payload: initial_payload_for_opt,
                allowed_mutable_bytes: allowed_mutable_bytes.clone(),
                enable_image_space_perturb: args.enable_image_space_perturb,
                progress_every_iters: if args.progress_every_iters == 0 {
                    None
                } else {
                    Some(args.progress_every_iters)
                },
                progress_every_secs: if args.progress_every_secs == 0 {
                    None
                } else {
                    Some(args.progress_every_secs)
                },
                progress_image_path: Some(frame_png.clone()),
                seed: frame_seed,
                max_iters: frame_max_iters,
                max_time_ms: Some(time_budget_ms),
                restarts: frame_restarts,
                single_thread: seeded_mode,
                ..OptimizeConfig::default()
            };
            let result = optimize(input, cfg)?;
            let canonical = encode_with_mask(spec, &result.best_payload, result.best_encoded.mask)?;
            write_qr_png(&canonical.modules, &frame_png)?;

            let report = RunReport {
                version: args.version,
                ecc: args.ecc.clone(),
                mask: match result.best_encoded.mask {
                    MaskPattern::M0 => 0,
                    MaskPattern::M1 => 1,
                    MaskPattern::M2 => 2,
                    MaskPattern::M3 => 3,
                    MaskPattern::M4 => 4,
                    MaskPattern::M5 => 5,
                    MaskPattern::M6 => 6,
                    MaskPattern::M7 => 7,
                },
                seed: frame_seed,
                capacity_bytes: result.best_payload.len(),
                prefix_len: prefix.len(),
                suffix_len: result.best_payload.len().saturating_sub(prefix.len()),
                iterations: result.iterations,
                restarts: result.restarts_done,
                elapsed_ms: result.elapsed_ms,
                ssim: result.best_score.ssim,
                mutable_match_pct: result.best_score.mutable_match_pct,
                full_match_pct: result.best_score.full_match_pct,
                objective_score: result.best_score.objective_score,
                decode_ok: None,
                decoded_payload_hex: None,
            };

            if args.write_json {
                let file = File::create(&frame_json).map_err(|e| {
                    QrodeError::Internal(format!("failed to create report json: {e}"))
                })?;
                serde_json::to_writer_pretty(file, &report).map_err(|e| {
                    QrodeError::Internal(format!("failed to write report json: {e}"))
                })?;
            }

            pb.inc(1);
            Ok(result.best_payload)
        };

        let mut idx = 0usize;
        let mut seed_cache: Vec<SeedCacheEntry> = Vec::new();
        let mut exact_reuse_cache: HashMap<u64, Vec<usize>> = HashMap::new();
        while idx < frame_jobs.len() {
            let current_target_hash = hash_target(&frame_jobs[idx].target);
            let mut exact_reuse_payload: Option<(usize, Vec<u8>)> = None;
            if let Some(entries) = exact_reuse_cache.get(&current_target_hash) {
                for &cache_pos in entries.iter().rev() {
                    let entry = &seed_cache[cache_pos];
                    if frame_jobs[idx].index.saturating_sub(entry.frame_index) <= 10 {
                        continue;
                    }
                    if entry.target == frame_jobs[idx].target {
                        exact_reuse_payload = Some((entry.frame_index, entry.payload.clone()));
                        break;
                    }
                }
            }

            if let Some((source_idx, payload)) = exact_reuse_payload {
                pb.println(format!(
                    "frame={} exact-reuse seed_from={} (older than 10)",
                    frame_jobs[idx].path.display(),
                    source_idx + 1,
                ));
                let payload =
                    run_frame(&frame_jobs[idx], Some(payload), Some(source_idx), Some(0.0))?;
                seed_cache.push(SeedCacheEntry {
                    frame_index: frame_jobs[idx].index,
                    target: frame_jobs[idx].target.clone(),
                    payload,
                });
                exact_reuse_cache
                    .entry(current_target_hash)
                    .or_default()
                    .push(seed_cache.len() - 1);
                idx += 1;
                continue;
            }

            let current_seed = nearest_seed_from_cache(&frame_jobs[idx].target, &seed_cache)?;
            let should_seed = current_seed
                .as_ref()
                .map(|(_, diff_ratio, _)| *diff_ratio <= frame_threshold)
                .unwrap_or(false);

            if !should_seed {
                let payload = run_frame(&frame_jobs[idx], None, None, None)?;
                seed_cache.push(SeedCacheEntry {
                    frame_index: frame_jobs[idx].index,
                    target: frame_jobs[idx].target.clone(),
                    payload,
                });
                exact_reuse_cache
                    .entry(current_target_hash)
                    .or_default()
                    .push(seed_cache.len() - 1);
                idx += 1;
                continue;
            }

            let start = idx;
            let mut batch: Vec<(usize, usize, f32, Vec<u8>)> = Vec::new();
            while idx < frame_jobs.len() {
                let candidate = nearest_seed_from_cache(&frame_jobs[idx].target, &seed_cache)?;
                let Some((source_idx, diff_ratio, payload)) = candidate else {
                    break;
                };
                if diff_ratio > frame_threshold {
                    break;
                }
                batch.push((idx, source_idx, diff_ratio, payload));
                idx += 1;
            }
            let end = start + batch.len();
            if batch.is_empty() {
                let payload = run_frame(&frame_jobs[start], None, None, None)?;
                seed_cache.push(SeedCacheEntry {
                    frame_index: frame_jobs[start].index,
                    target: frame_jobs[start].target.clone(),
                    payload,
                });
                exact_reuse_cache
                    .entry(hash_target(&frame_jobs[start].target))
                    .or_default()
                    .push(seed_cache.len() - 1);
                idx = start + 1;
                continue;
            }

            pb.println(format!(
                "parallel seeded batch: frames {}..{}",
                start + 1,
                end
            ));

            let payloads_with_idx: Vec<(usize, Vec<u8>)> = batch
                .par_iter()
                .map(|(job_idx, source_idx, diff_ratio, payload)| {
                    run_frame(
                        &frame_jobs[*job_idx],
                        Some(payload.clone()),
                        Some(*source_idx),
                        Some(*diff_ratio),
                    )
                    .map(|best_payload| (*job_idx, best_payload))
                })
                .collect::<Result<Vec<_>>>()?;

            for (job_idx, payload) in payloads_with_idx {
                let job = &frame_jobs[job_idx];
                seed_cache.push(SeedCacheEntry {
                    frame_index: job.index,
                    target: job.target.clone(),
                    payload,
                });
                exact_reuse_cache
                    .entry(hash_target(&job.target))
                    .or_default()
                    .push(seed_cache.len() - 1);
            }
        }

        pb.finish_with_message("frames complete");

        return Ok(());
    }

    let target = if is_circle_target {
        synthetic_circle_target(args.version)?
    } else {
        let target_path = args.target.ok_or_else(|| {
            QrodeError::Internal("--target is required when --target-kind is not circle".into())
        })?;
        load_and_adapt_target(
            &target_path,
            args.version,
            TargetAdapterConfig {
                fit_mode: FitMode::Contain,
                threshold: args.threshold,
                invert: false,
            },
        )?
    };

    if is_circle_target {
        let target_png = derived_target_out_path(&args.out_png);
        write_bitmap_png(&target, &target_png)?;
        println!("wrote circle target bitmap: {}", target_png.display());
    }

    let cfg = OptimizeConfig {
        spec,
        fixed_mask,
        search_masks: args.search_masks,
        initial_payload: None,
        allowed_mutable_bytes: allowed_mutable_bytes.clone(),
        enable_image_space_perturb: args.enable_image_space_perturb,
        progress_every_iters: if args.progress_every_iters == 0 {
            None
        } else {
            Some(args.progress_every_iters)
        },
        progress_every_secs: if args.progress_every_secs == 0 {
            None
        } else {
            Some(args.progress_every_secs)
        },
        progress_image_path: Some(args.out_png.clone()),
        seed,
        max_iters: args.max_iters.unwrap_or(default_iters),
        max_time_ms: Some(time_budget_ms),
        restarts: args.restarts.unwrap_or(default_restarts),
        single_thread: false,
        ..OptimizeConfig::default()
    };
    let input = OptimizeInput {
        prefix: prefix.clone(),
        target,
        weights: ScoreWeights {
            use_ssim: args.use_ssim,
            ..ScoreWeights::default()
        },
    };

    let result = optimize(input, cfg)?;
    let canonical = encode_with_mask(spec, &result.best_payload, result.best_encoded.mask)?;
    write_qr_png(&canonical.modules, &args.out_png)?;

    let validation = if args.benchmark_decode {
        Some(validate_with_decoder(&args.out_png, &result.best_payload)?)
    } else {
        None
    };

    let report = RunReport {
        version: args.version,
        ecc: args.ecc,
        mask: match result.best_encoded.mask {
            MaskPattern::M0 => 0,
            MaskPattern::M1 => 1,
            MaskPattern::M2 => 2,
            MaskPattern::M3 => 3,
            MaskPattern::M4 => 4,
            MaskPattern::M5 => 5,
            MaskPattern::M6 => 6,
            MaskPattern::M7 => 7,
        },
        seed,
        capacity_bytes: result.best_payload.len(),
        prefix_len: prefix.len(),
        suffix_len: result.best_payload.len().saturating_sub(prefix.len()),
        iterations: result.iterations,
        restarts: result.restarts_done,
        elapsed_ms: result.elapsed_ms,
        ssim: result.best_score.ssim,
        mutable_match_pct: result.best_score.mutable_match_pct,
        full_match_pct: result.best_score.full_match_pct,
        objective_score: result.best_score.objective_score,
        decode_ok: validation.as_ref().map(|v| v.decode_ok),
        decoded_payload_hex: validation
            .as_ref()
            .and_then(|v| v.decoded_payload.as_ref())
            .map(|bytes| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()),
    };

    if args.write_json {
        let file = File::create(&args.out_json)
            .map_err(|e| QrodeError::Internal(format!("failed to create report json: {e}")))?;
        serde_json::to_writer_pretty(file, &report)
            .map_err(|e| QrodeError::Internal(format!("failed to write report json: {e}")))?;
    }

    println!(
        "best mutable={:.3}% full={:.3}% objective={:.6} elapsed={}ms",
        report.mutable_match_pct, report.full_match_pct, report.objective_score, report.elapsed_ms
    );
    if let Some(v) = validation {
        println!("decoder validation: {} ({})", v.decode_ok, v.message);
    }
    Ok(())
}
