use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::qr_core::{
	EncodedQr, MaskPattern, QrSpec, QrodeError, Result, capacity_bytes, encode_with_mask, immutable_mask,
};
use crate::scoring::{ScoreBreakdown, ScoreWeights, score_modules};
use crate::validator::decode_modules_payload;

#[derive(Clone, Debug)]
pub struct OptimizeConfig {
	pub spec: QrSpec,
	pub fixed_mask: Option<MaskPattern>,
	pub search_masks: bool,
	pub initial_payload: Option<Vec<u8>>,
	pub allowed_mutable_bytes: Option<Vec<u8>>,
	pub enable_image_space_perturb: bool,
	pub image_space_perturb_iters: u64,
	pub progress_every_iters: Option<u64>,
	pub progress_every_secs: Option<u64>,
	pub progress_image_path: Option<PathBuf>,
	pub bitwise_seed_init: bool,
	pub single_thread: bool,
	pub coordinate_refine_sweeps: u32,
	pub max_time_ms: Option<u128>,
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
				ecc: crate::qr_core::EccLevel::H,
				mode: crate::qr_core::DataMode::Byte,
			},
			fixed_mask: Some(MaskPattern::M0),
			search_masks: false,
			initial_payload: None,
			allowed_mutable_bytes: None,
			enable_image_space_perturb: true,
			image_space_perturb_iters: 2_000,
			progress_every_iters: None,
			progress_every_secs: Some(10),
			progress_image_path: None,
			bitwise_seed_init: true,
			single_thread: false,
			coordinate_refine_sweeps: 3,
			max_time_ms: None,
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

struct ProgressState {
	best_objective: f32,
	output_path: PathBuf,
}

fn weighted_index(weights: &[f32], rng: &mut Xoshiro256PlusPlus) -> usize {
	let total: f32 = weights.iter().copied().sum();
	if total <= f32::EPSILON {
		return rng.gen_range(0..weights.len());
	}
	let mut r = rng.gen_range(0.0..total);
	for (i, &w) in weights.iter().enumerate() {
		r -= w.max(0.0);
		if r <= 0.0 {
			return i;
		}
	}
	weights.len() - 1
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

fn random_mutable_byte(cfg: &OptimizeConfig, rng: &mut Xoshiro256PlusPlus) -> u8 {
	if let Some(allowed) = cfg.allowed_mutable_bytes.as_ref() {
		allowed[rng.gen_range(0..allowed.len())]
	} else {
		rng.r#gen()
	}
}

fn write_progress_png(modules: &[Vec<bool>], out_png: &PathBuf) -> Result<()> {
	let module_px = 4u32;
	let quiet_zone = 4u32;
	let n = modules.len() as u32;
	let img_size = (n + 2 * quiet_zone) * module_px;

	let mut img = image::GrayImage::from_pixel(img_size, img_size, image::Luma([255u8]));
	for (y, row) in modules.iter().enumerate() {
		for (x, &dark) in row.iter().enumerate() {
			let color = if dark { 0u8 } else { 255u8 };
			let px0 = (x as u32 + quiet_zone) * module_px;
			let py0 = (y as u32 + quiet_zone) * module_px;
			for py in py0..py0 + module_px {
				for px in px0..px0 + module_px {
					img.put_pixel(px, py, image::Luma([color]));
				}
			}
		}
	}

	img.save(out_png)
		.map_err(|e| QrodeError::Internal(format!("failed to save progress png: {e}")))
}

fn maybe_emit_progress(
	cfg: &OptimizeConfig,
	restart_idx: u32,
	iter: u64,
	best_score: ScoreBreakdown,
	best_enc: &EncodedQr,
	restart_started: Instant,
	last_progress_emit: &mut Instant,
	progress_state: Option<&Arc<Mutex<ProgressState>>>,
) -> Result<()> {
	let iter_trigger = cfg
		.progress_every_iters
		.map(|every| every > 0 && (iter + 1) % every == 0)
		.unwrap_or(false);
	let time_trigger = cfg
		.progress_every_secs
		.map(|secs| secs > 0 && last_progress_emit.elapsed().as_secs() >= secs)
		.unwrap_or(false);

	if !iter_trigger && !time_trigger {
		return Ok(());
	}

	println!(
		"progress restart={} iter={} objective={:.6} ssim={:.6} mutable={:.3}% full={:.3}% elapsed={}ms",
		restart_idx,
		iter + 1,
		best_score.objective_score,
		best_score.ssim,
		best_score.mutable_match_pct,
		best_score.full_match_pct,
		restart_started.elapsed().as_millis()
	);
	let _ = std::io::stdout().flush();
	*last_progress_emit = Instant::now();

	if let Some(state) = progress_state {
		let mut guard = state
			.lock()
			.map_err(|_| QrodeError::Internal("progress state lock poisoned".into()))?;
		if best_score.objective_score > guard.best_objective {
			guard.best_objective = best_score.objective_score;
			write_progress_png(&best_enc.modules, &guard.output_path)?;
		}
	}

	Ok(())
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

fn mutate_payload(
	candidate: &mut [u8],
	prefix_len: usize,
	cfg: &OptimizeConfig,
	influence: &[f32],
	rng: &mut Xoshiro256PlusPlus,
) -> Vec<usize> {
	if prefix_len >= candidate.len() {
		return Vec::new();
	}
	let start = prefix_len;
	let len = candidate.len() - prefix_len;
	if len == 0 {
		return Vec::new();
	}
	let mut touched = Vec::with_capacity(8);

	if rng.gen_bool(cfg.block_mutation_prob as f64) {
		let block_len = rng.gen_range(2..=len.min(16));
		let block_start = rng.gen_range(start..=(candidate.len() - block_len));
		for (offset, b) in candidate[block_start..block_start + block_len].iter_mut().enumerate() {
			*b = random_mutable_byte(cfg, rng);
			touched.push(block_start + offset);
		}
		return touched;
	}

	if rng.gen_bool(cfg.byte_mutation_rate as f64) {
		let idx = rng.gen_range(start..candidate.len());
		candidate[idx] = random_mutable_byte(cfg, rng);
		touched.push(idx);
	}

	if cfg.allowed_mutable_bytes.is_none() {
		if rng.gen_bool(cfg.bit_mutation_rate as f64) {
			let idx = rng.gen_range(start..candidate.len());
			let bit = rng.gen_range(0..8);
			candidate[idx] ^= 1u8 << bit;
			touched.push(idx);
		}

		if rng.gen_bool(cfg.guided_mutation_prob as f64) {
			let idx = start + weighted_index(influence, rng);
			let bit = rng.gen_range(0..8);
			candidate[idx] ^= 1u8 << bit;
			touched.push(idx);
		}
	} else if rng.gen_bool(cfg.guided_mutation_prob as f64) {
		let idx = start + weighted_index(influence, rng);
		candidate[idx] = random_mutable_byte(cfg, rng);
		touched.push(idx);
	}

	if touched.is_empty() {
		let idx = rng.gen_range(start..candidate.len());
		if cfg.allowed_mutable_bytes.is_none() {
			let bit = rng.gen_range(0..8);
			candidate[idx] ^= 1u8 << bit;
		} else {
			candidate[idx] = random_mutable_byte(cfg, rng);
		}
		touched.push(idx);
	}

	touched.sort_unstable();
	touched.dedup();
	touched
}

fn run_single_restart(
	input: &OptimizeInput,
	cfg: &OptimizeConfig,
	eval_ctx: &EvalContext,
	seed_payload: Option<&[u8]>,
	progress_state: Option<&Arc<Mutex<ProgressState>>>,
	restart_idx: u32,
	seed: u64,
) -> Result<(Vec<u8>, EncodedQr, ScoreBreakdown)> {
	let restart_started = Instant::now();
	let stop_at = cfg
		.max_time_ms
		.map(|budget| restart_started + std::time::Duration::from_millis(budget as u64));

	let cap = capacity_bytes(cfg.spec)?;
	if input.prefix.len() > cap {
		return Err(QrodeError::PrefixTooLong {
			prefix_len: input.prefix.len(),
			capacity: cap,
		});
	}

	let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
	let mut current = if let Some(seed_payload) = seed_payload {
		let mut seeded = seed_payload.to_vec();
		// Diversify restarts by perturbing the deterministic seed.
		if restart_idx > 0 && input.prefix.len() < seeded.len() {
			let available = seeded.len() - input.prefix.len();
			let changes = usize::min(available, (restart_idx as usize * 2).min(24));
			for _ in 0..changes {
				let idx = rng.gen_range(input.prefix.len()..seeded.len());
				if cfg.allowed_mutable_bytes.is_none() {
					let bit = rng.gen_range(0..8);
					seeded[idx] ^= 1u8 << bit;
				} else {
					seeded[idx] = random_mutable_byte(cfg, &mut rng);
				}
			}
		}
		seeded
	} else {
		let mut random_init = vec![0u8; cap];
		random_init[..input.prefix.len()].copy_from_slice(&input.prefix);
		for b in &mut random_init[input.prefix.len()..] {
			*b = random_mutable_byte(cfg, &mut rng);
		}
		random_init
	};

	current[..input.prefix.len()].copy_from_slice(&input.prefix);

	let (mut current_enc, mut current_score) = evaluate_payload(&current, cfg, input, eval_ctx)?;
	let mut best_payload = current.clone();
	let mut best_enc = current_enc.clone();
	let mut best_score = current_score;
	let mut influence = vec![1.0f32; cap - input.prefix.len()];
	let mut since_improvement = 0u64;
	let mut last_progress_emit = restart_started;

	for iter in 0..cfg.max_iters {
		if let Some(deadline) = stop_at {
			if Instant::now() >= deadline {
				break;
			}
		}

		let mut proposal = current.clone();
		let touched = mutate_payload(&mut proposal, input.prefix.len(), cfg, &influence, &mut rng);
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
			if delta > 0.0 {
				for idx in touched {
					let local = idx - input.prefix.len();
					if let Some(weight) = influence.get_mut(local) {
						*weight += 1.0 + delta * 250.0;
					}
				}
			}
		}

		if current_score.objective_score > best_score.objective_score {
			best_payload = current.clone();
			best_enc = current_enc.clone();
			best_score = current_score;
			since_improvement = 0;
		} else {
			since_improvement += 1;
		}

		if (iter + 1) % 512 == 0 {
			for w in &mut influence {
				*w = (*w * 0.995).max(0.1);
			}
		}

		if since_improvement > 600 && input.prefix.len() < current.len() {
			since_improvement = 0;
			current = best_payload.clone();
			let available = current.len() - input.prefix.len();
			let span = available.min(24);
			let min_burst = 1usize.min(span);
			let burst = rng.gen_range(min_burst..=span);
			let start = rng.gen_range(input.prefix.len()..=(current.len() - burst));
			for b in &mut current[start..start + burst] {
				*b = random_mutable_byte(cfg, &mut rng);
			}
			let (enc, score) = evaluate_payload(&current, cfg, input, eval_ctx)?;
			current_enc = enc;
			current_score = score;
		}

		maybe_emit_progress(
			cfg,
			restart_idx,
			iter,
			best_score,
			&best_enc,
			restart_started,
			&mut last_progress_emit,
			progress_state,
		)?;
	}

	let (best_payload, best_enc, best_score) = coordinate_refine(
		best_payload,
		best_enc,
		best_score,
		input,
		cfg,
		eval_ctx,
		&mut rng,
		stop_at,
	)?;
	let (best_enc, best_score) = image_space_perturb(
		best_enc,
		best_score,
		input,
		cfg,
		eval_ctx,
		&mut rng,
		stop_at,
	)?;

	Ok((best_payload, best_enc, best_score))
}

fn build_bitwise_seed_payload(
	input: &OptimizeInput,
	cfg: &OptimizeConfig,
	eval_ctx: &EvalContext,
) -> Result<Vec<u8>> {
	let cap = capacity_bytes(cfg.spec)?;
	if input.prefix.len() > cap {
		return Err(QrodeError::PrefixTooLong {
			prefix_len: input.prefix.len(),
			capacity: cap,
		});
	}

	let mut current = vec![0u8; cap];
	current[..input.prefix.len()].copy_from_slice(&input.prefix);
	let (_, mut current_score) = evaluate_payload(&current, cfg, input, eval_ctx)?;

	if let Some(allowed) = cfg.allowed_mutable_bytes.as_ref() {
		for byte_idx in input.prefix.len()..cap {
			let mut best_byte = current[byte_idx];
			let mut best_local_score = current_score;
			for candidate in allowed {
				if *candidate == current[byte_idx] {
					continue;
				}
				let mut proposal = current.clone();
				proposal[byte_idx] = *candidate;
				let (_, proposal_score) = evaluate_payload(&proposal, cfg, input, eval_ctx)?;
				if proposal_score.objective_score > best_local_score.objective_score {
					best_byte = *candidate;
					best_local_score = proposal_score;
				}
			}
			if best_byte != current[byte_idx] {
				current[byte_idx] = best_byte;
				current_score = best_local_score;
			}
		}
		return Ok(current);
	}

	for bit_idx in (input.prefix.len() * 8)..(cap * 8) {
		let byte_idx = bit_idx / 8;
		let bit = (bit_idx % 8) as u8;
		let mut proposal = current.clone();
		proposal[byte_idx] ^= 1u8 << bit;

		let (_, proposal_score) = evaluate_payload(&proposal, cfg, input, eval_ctx)?;
		if proposal_score.objective_score > current_score.objective_score {
			current = proposal;
			current_score = proposal_score;
		}
	}

	Ok(current)
}

fn coordinate_refine(
	mut payload: Vec<u8>,
	mut enc: EncodedQr,
	mut score: ScoreBreakdown,
	input: &OptimizeInput,
	cfg: &OptimizeConfig,
	eval_ctx: &EvalContext,
	rng: &mut Xoshiro256PlusPlus,
	stop_at: Option<Instant>,
) -> Result<(Vec<u8>, EncodedQr, ScoreBreakdown)> {
	if cfg.coordinate_refine_sweeps == 0 || input.prefix.len() >= payload.len() {
		return Ok((payload, enc, score));
	}

	if let Some(allowed) = cfg.allowed_mutable_bytes.as_ref() {
		let mut indices: Vec<usize> = (input.prefix.len()..payload.len()).collect();
		for _ in 0..cfg.coordinate_refine_sweeps {
			indices.shuffle(rng);
			let mut improved = false;

			for byte_idx in &indices {
				if let Some(deadline) = stop_at {
					if Instant::now() >= deadline {
						return Ok((payload, enc, score));
					}
				}

				let mut best_byte = payload[*byte_idx];
				let mut best_local_score = score;
				let mut best_local_enc: Option<EncodedQr> = None;

				for candidate in allowed {
					if *candidate == payload[*byte_idx] {
						continue;
					}
					let mut proposal = payload.clone();
					proposal[*byte_idx] = *candidate;
					let (proposal_enc, proposal_score) = evaluate_payload(&proposal, cfg, input, eval_ctx)?;
					if proposal_score.objective_score > best_local_score.objective_score {
						best_byte = *candidate;
						best_local_score = proposal_score;
						best_local_enc = Some(proposal_enc);
					}
				}

				if best_byte != payload[*byte_idx] {
					payload[*byte_idx] = best_byte;
					score = best_local_score;
					enc = best_local_enc.expect("best_local_enc present when improvement found");
					improved = true;
				}
			}

			if !improved {
				break;
			}
		}

		return Ok((payload, enc, score));
	}

	let mutable_start = input.prefix.len() * 8;
	let mutable_end = payload.len() * 8;
	let mut indices: Vec<usize> = (mutable_start..mutable_end).collect();

	for _ in 0..cfg.coordinate_refine_sweeps {
		indices.shuffle(rng);
		let mut improved = false;

		for bit_idx in &indices {
			if let Some(deadline) = stop_at {
				if Instant::now() >= deadline {
					return Ok((payload, enc, score));
				}
			}

			let byte_idx = bit_idx / 8;
			let bit = (bit_idx % 8) as u8;
			let mut proposal = payload.clone();
			proposal[byte_idx] ^= 1u8 << bit;

			let (proposal_enc, proposal_score) = evaluate_payload(&proposal, cfg, input, eval_ctx)?;
			if proposal_score.objective_score > score.objective_score {
				payload = proposal;
				enc = proposal_enc;
				score = proposal_score;
				improved = true;
			}
		}

		if !improved {
			break;
		}
	}

	Ok((payload, enc, score))
}

fn immutable_mask_for_selected_mask<'a>(
	eval_ctx: &'a EvalContext,
	selected: MaskPattern,
) -> Option<&'a Vec<Vec<bool>>> {
	eval_ctx
		.masks
		.iter()
		.find_map(|(mask, imm)| if *mask == selected { Some(imm) } else { None })
}

fn image_space_perturb(
	mut enc: EncodedQr,
	mut score: ScoreBreakdown,
	input: &OptimizeInput,
	cfg: &OptimizeConfig,
	eval_ctx: &EvalContext,
	rng: &mut Xoshiro256PlusPlus,
	stop_at: Option<Instant>,
) -> Result<(EncodedQr, ScoreBreakdown)> {
	if !cfg.enable_image_space_perturb || cfg.image_space_perturb_iters == 0 {
		return Ok((enc, score));
	}

	let Some(imm) = immutable_mask_for_selected_mask(eval_ctx, enc.mask) else {
		return Ok((enc, score));
	};

	let n = enc.modules.len();
	if n == 0 {
		return Ok((enc, score));
	}

	let mut coords = Vec::new();
	for y in 0..n {
		for x in 0..n {
			if !imm[y][x] {
				coords.push((x, y));
			}
		}
	}
	if coords.is_empty() {
		return Ok((enc, score));
	}

	let cap = capacity_bytes(cfg.spec)?;
	for _ in 0..cfg.image_space_perturb_iters {
		if let Some(deadline) = stop_at {
			if Instant::now() >= deadline {
				break;
			}
		}

		let (x, y) = coords[rng.gen_range(0..coords.len())];
		enc.modules[y][x] = !enc.modules[y][x];

		let proposal = score_modules(&enc.modules, &input.target, imm, input.weights)?;
		if proposal.objective_score > score.objective_score {
			if let Some(decoded) = decode_modules_payload(&enc.modules) {
				if decoded.len() == cap && decoded.starts_with(&input.prefix) {
					score = proposal;
					continue;
				}
			}
		}

		// Reject perturbation if no improvement or decode constraints are violated.
		enc.modules[y][x] = !enc.modules[y][x];
	}

	Ok((enc, score))
}

pub fn optimize(input: OptimizeInput, config: OptimizeConfig) -> Result<OptimizeResult> {
	let started = Instant::now();
	if let Some(allowed) = config.allowed_mutable_bytes.as_ref() {
		if allowed.is_empty() {
			return Err(QrodeError::Internal(
				"allowed_mutable_bytes cannot be empty".into(),
			));
		}
	}
	let eval_ctx = build_eval_context(&config)?;
	let cap = capacity_bytes(config.spec)?;
	let seed_payload = if let Some(initial) = config.initial_payload.as_ref() {
		if initial.len() != cap {
			return Err(QrodeError::Internal(format!(
				"initial_payload length mismatch: got {}, expected {}",
				initial.len(),
				cap,
			)));
		}
		if !initial.starts_with(&input.prefix) {
			return Err(QrodeError::Internal(
				"initial_payload does not preserve required prefix".into(),
			));
		}
		Some(initial.clone())
	} else if config.bitwise_seed_init {
		Some(build_bitwise_seed_payload(&input, &config, &eval_ctx)?)
	} else {
		None
	};
	let progress_state = config.progress_image_path.as_ref().map(|path| {
		Arc::new(Mutex::new(ProgressState {
			best_objective: f32::NEG_INFINITY,
			output_path: path.clone(),
		}))
	});

	let restarts = config.restarts.max(1);
	let results: Vec<Result<(Vec<u8>, EncodedQr, ScoreBreakdown)>> = if config.single_thread {
		(0..restarts)
			.map(|restart| {
				let progress_state = progress_state.clone();
				let seed = config.seed.wrapping_add(restart as u64 * 1_000_003);
				run_single_restart(
					&input,
					&config,
					&eval_ctx,
					seed_payload.as_deref(),
					progress_state.as_ref(),
					restart,
					seed,
				)
			})
			.collect()
	} else {
		(0..restarts)
			.into_par_iter()
			.map(|restart| {
				let progress_state = progress_state.clone();
				let seed = config.seed.wrapping_add(restart as u64 * 1_000_003);
				run_single_restart(
					&input,
					&config,
					&eval_ctx,
					seed_payload.as_deref(),
					progress_state.as_ref(),
					restart,
					seed,
				)
			})
			.collect()
	};

	let mut global_best: Option<(Vec<u8>, EncodedQr, ScoreBreakdown)> = None;
	for result in results {
		let result = result?;
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
		iterations: config.max_iters * restarts as u64,
		restarts_done: restarts,
		elapsed_ms: started.elapsed().as_millis(),
		seed: config.seed,
	})
}

pub fn bench_hot_loop(iterations: u64, version: u8, seed: u64) -> Result<f32> {
	let spec = QrSpec {
		version,
		ecc: crate::qr_core::EccLevel::H,
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
