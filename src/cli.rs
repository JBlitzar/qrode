use image::{ImageBuffer, Luma};
use serde::Serialize;
use std::fs::File;
use std::path::PathBuf;

use crate::optimizer::{OptimizeConfig, OptimizeInput, optimize};
use crate::qr_core::{DataMode, EccLevel, MaskPattern, QrSpec, QrodeError, Result, capacity_bytes};
use crate::scoring::ScoreWeights;
use crate::target_adapter::{FitMode, TargetAdapterConfig, load_and_adapt_target, synthetic_circle_target};
use crate::validator::{validate_payload_policy, validate_with_decoder};

const URL_MODE_PREFIX: &str = "https://example.com?";

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum PayloadMode {
	Binary,
	Url,
}

#[derive(clap::Parser, Debug)]
pub struct CliArgs {
	#[arg(long)]
	pub target: Option<PathBuf>,
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
	#[arg(long, default_value_t = false)]
	pub search_masks: bool,
	#[arg(long = "no-image-space-perturb", action = clap::ArgAction::SetFalse, default_value_t = true)]
	pub enable_image_space_perturb: bool,
	#[arg(long = "no-ssim", action = clap::ArgAction::SetFalse, default_value_t = true)]
	pub use_ssim: bool,
	#[arg(long, default_value_t = false)]
	pub benchmark_decode: bool,
	#[arg(long, default_value_t = 128)]
	pub threshold: u8,
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

fn parse_prefix_hex(s: &str) -> Result<Vec<u8>> {
	if s.is_empty() {
		return Ok(Vec::new());
	}
	if !s.len().is_multiple_of(2) {
		return Err(QrodeError::Internal("prefix_hex must have even length".into()));
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

	let mut img: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_pixel(img_size, img_size, Luma([255]));
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

	let mut img: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_pixel(img_size, img_size, Luma([255]));
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
	let parent = out_png.parent().unwrap_or_else(|| std::path::Path::new("."));
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
		_ => return Err(QrodeError::Internal("invalid ECC level. Expected H, L, M, or Q".into())),
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

	let (default_iters, default_restarts, default_time_budget_secs) = match args.mode.as_str() {
		"quick" => (5_000_000, (rayon::current_num_threads() as u32).clamp(2, 8), 15u64),
		"quality" => (50_000_000, ((rayon::current_num_threads() - 1) as u32).clamp(2, 16), 55u64),
		_ => return Err(QrodeError::Internal("mode must be quick or quality".into())),
	};
	let seed = args.seed.unwrap_or(1);
	let time_budget_ms = args
		.time_budget_secs
		.unwrap_or(default_time_budget_secs)
		.saturating_mul(1000) as u128;

	let fixed_mask = args.mask.map(|m| {
		MaskPattern::from_u8(m).ok_or_else(|| QrodeError::Internal("mask must be 0..7".into()))
	}).transpose()?;

	let cfg = OptimizeConfig {
		spec,
		fixed_mask,
		search_masks: args.search_masks,
		allowed_mutable_bytes,
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
	write_qr_png(&result.best_encoded.modules, &args.out_png)?;

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

	let file = File::create(&args.out_json)
		.map_err(|e| QrodeError::Internal(format!("failed to create report json: {e}")))?;
	serde_json::to_writer_pretty(file, &report)
		.map_err(|e| QrodeError::Internal(format!("failed to write report json: {e}")))?;

	println!(
		"best mutable={:.3}% full={:.3}% objective={:.6} elapsed={}ms",
		report.mutable_match_pct, report.full_match_pct, report.objective_score, report.elapsed_ms
	);
	if let Some(v) = validation {
		println!("decoder validation: {} ({})", v.decode_ok, v.message);
	}
	Ok(())
}
