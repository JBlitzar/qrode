use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Instant;

use crate::qr_core::{
	EncodedQr, MaskPattern, QrSpec, QrodeError, Result, capacity_bytes, encode_with_mask, immutable_mask,
};
use crate::scoring::{ScoreBreakdown, ScoreWeights, score_modules};

#[derive(Clone, Debug)]
pub struct OptimizeConfig {
	pub spec: QrSpec,
	pub fixed_mask: Option<MaskPattern>,
	pub search_masks: bool,
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

impl Default for OptimizeConfig {
	fn default() -> Self {
		Self {
			spec: QrSpec {
				version: 5,
				ecc: crate::qr_core::EccLevel::M,
				mode: crate::qr_core::DataMode::Byte,
			},
			fixed_mask: Some(MaskPattern::M0),
			search_masks: false,
			seed: 1,
			max_iters: 10_000,
			restarts: 3,
			init_temp: 0.05,
			final_temp: 0.001,
			byte_mutation_rate: 0.3,
			bit_mutation_rate: 0.5,
			block_mutation_prob: 0.2,
			guided_mutation_prob: 0.1,
		}
	}
}

#[derive(Clone, Debug)]
pub struct OptimizeInput {
	pub prefix: Vec<u8>,
	pub target: Vec<Vec<bool>>,
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

struct EvalContext {
	masks: Vec<(MaskPattern, Vec<Vec<bool>>)>,
}

fn anneal_temp(cfg: &OptimizeConfig, iter: u64) -> f32 {
	if cfg.max_iters <= 1 {
		return cfg.final_temp.max(1e-9);
	}
	let t = iter as f32 / (cfg.max_iters - 1) as f32;
	let a = cfg.init_temp.max(1e-9).ln();
	let b = cfg.final_temp.max(1e-9).ln();
	((1.0 - t) * a + t * b).exp()
}

fn build_eval_context(cfg: &OptimizeConfig) -> Result<EvalContext> {
	let masks = if cfg.search_masks {
		let mut out = Vec::with_capacity(8);
		for mask in MaskPattern::all() {
			out.push((mask, immutable_mask(cfg.spec, mask)?));
		}
		out
	} else {
		let mask = cfg.fixed_mask.unwrap_or(MaskPattern::M0);
		vec![(mask, immutable_mask(cfg.spec, mask)?)]
	};
	Ok(EvalContext { masks })
}

fn evaluate_payload(
	payload: &[u8],
	cfg: &OptimizeConfig,
	input: &OptimizeInput,
	eval_ctx: &EvalContext,
) -> Result<(EncodedQr, ScoreBreakdown)> {
	let mut best: Option<(EncodedQr, ScoreBreakdown)> = None;

	for (mask, imm) in &eval_ctx.masks {
		let enc = encode_with_mask(cfg.spec, payload, *mask)?;
		let score = score_modules(&enc.modules, &input.target, imm, input.weights)?;

		match &best {
			Some((_, current)) if current.objective_score >= score.objective_score => {}
			_ => best = Some((enc, score)),
		}
	}

	best.ok_or_else(|| QrodeError::Internal("no mask candidates available".into()))
}

fn mutate_payload(candidate: &mut [u8], prefix_len: usize, cfg: &OptimizeConfig, rng: &mut Xoshiro256PlusPlus) {
	if prefix_len >= candidate.len() {
		return;
	}
	let start = prefix_len;
	let len = candidate.len() - prefix_len;

	if rng.gen_bool(cfg.block_mutation_prob as f64) {
		let block_len = rng.gen_range(1..=len.min(8));
		let block_start = rng.gen_range(start..=(candidate.len() - block_len));
		for b in &mut candidate[block_start..block_start + block_len] {
			*b = rng.r#gen();
		}
		return;
	}

	if rng.gen_bool(cfg.byte_mutation_rate as f64) {
		let idx = rng.gen_range(start..candidate.len());
		candidate[idx] = rng.r#gen();
	}

	if rng.gen_bool(cfg.bit_mutation_rate as f64) {
		let idx = rng.gen_range(start..candidate.len());
		let bit = rng.gen_range(0..8);
		candidate[idx] ^= 1u8 << bit;
	}

	if rng.gen_bool(cfg.guided_mutation_prob as f64) {
		let idx = rng.gen_range(start..candidate.len());
		let tweak = if rng.gen_bool(0.5) { 0x0f } else { 0xf0 };
		candidate[idx] ^= tweak;
	}
}

fn run_single_restart(
	input: &OptimizeInput,
	cfg: &OptimizeConfig,
	eval_ctx: &EvalContext,
	seed: u64,
) -> Result<(Vec<u8>, EncodedQr, ScoreBreakdown)> {
	let cap = capacity_bytes(cfg.spec)?;
	if input.prefix.len() > cap {
		return Err(QrodeError::PrefixTooLong {
			prefix_len: input.prefix.len(),
			capacity: cap,
		});
	}

	let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
	let mut current = vec![0u8; cap];
	current[..input.prefix.len()].copy_from_slice(&input.prefix);
	for b in &mut current[input.prefix.len()..] {
		*b = rng.r#gen();
	}

	let (mut current_enc, mut current_score) = evaluate_payload(&current, cfg, input, eval_ctx)?;
	let mut best_payload = current.clone();
	let mut best_enc = current_enc.clone();
	let mut best_score = current_score;

	for iter in 0..cfg.max_iters {
		let mut proposal = current.clone();
		mutate_payload(&mut proposal, input.prefix.len(), cfg, &mut rng);
		let (proposal_enc, proposal_score) = evaluate_payload(&proposal, cfg, input, eval_ctx)?;

		let delta = proposal_score.objective_score - current_score.objective_score;
		let accept = if delta >= 0.0 {
			true
		} else {
			let temp = anneal_temp(cfg, iter).max(1e-9);
			let prob = (delta / temp).exp().clamp(0.0, 1.0);
			rng.gen_bool(prob as f64)
		};

		if accept {
			current = proposal;
			current_enc = proposal_enc;
			current_score = proposal_score;
		}

		if current_score.objective_score > best_score.objective_score {
			best_payload = current.clone();
			best_enc = current_enc.clone();
			best_score = current_score;
		}
	}

	Ok((best_payload, best_enc, best_score))
}

pub fn optimize(input: OptimizeInput, config: OptimizeConfig) -> Result<OptimizeResult> {
	let started = Instant::now();
	let eval_ctx = build_eval_context(&config)?;

	let mut global_best: Option<(Vec<u8>, EncodedQr, ScoreBreakdown)> = None;
	for restart in 0..config.restarts.max(1) {
		let seed = config.seed.wrapping_add(restart as u64 * 1_000_003);
		let result = run_single_restart(&input, &config, &eval_ctx, seed)?;
		match &global_best {
			Some((_, _, score)) if score.objective_score >= result.2.objective_score => {}
			_ => global_best = Some(result),
		}
	}

	let (best_payload, best_encoded, best_score) =
		global_best.ok_or_else(|| QrodeError::Internal("optimizer produced no result".into()))?;

	Ok(OptimizeResult {
		best_payload,
		best_encoded,
		best_score,
		iterations: config.max_iters * config.restarts.max(1) as u64,
		restarts_done: config.restarts.max(1),
		elapsed_ms: started.elapsed().as_millis(),
		seed: config.seed,
	})
}

pub fn bench_hot_loop(iterations: u64, version: u8, seed: u64) -> Result<f32> {
	let spec = QrSpec {
		version,
		ecc: crate::qr_core::EccLevel::M,
		mode: crate::qr_core::DataMode::Byte,
	};
	let target = crate::target_adapter::synthetic_circle_target(version)?;
	let input = OptimizeInput {
		prefix: Vec::new(),
		target,
		weights: ScoreWeights::default(),
	};
	let cfg = OptimizeConfig {
		spec,
		seed,
		max_iters: iterations,
		restarts: 1,
		..OptimizeConfig::default()
	};
	let res = optimize(input, cfg)?;
	Ok(res.best_score.objective_score)
}
