use image::imageops::FilterType;
use std::path::Path;

use crate::qr_core::{Matrix, QrodeError, Result, module_size};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FitMode {
	Contain,
	Cover,
}

#[derive(Clone, Copy, Debug)]
pub struct TargetAdapterConfig {
	pub fit_mode: FitMode,
	pub threshold: u8,
	pub invert: bool,
}

impl Default for TargetAdapterConfig {
	fn default() -> Self {
		Self {
			fit_mode: FitMode::Contain,
			threshold: 128,
			invert: false,
		}
	}
}

pub fn load_and_adapt_target(
	image_path: &Path,
	version: u8,
	config: TargetAdapterConfig,
) -> Result<Matrix> {
	let n = module_size(version)?;
	let img = image::open(image_path).map_err(|e| QrodeError::Image(e.to_string()))?;
	let gray = img.to_luma8();

	let (w, h) = gray.dimensions();
	let n_u32 = n as u32;
	let scale = match config.fit_mode {
		FitMode::Contain => (n_u32 as f32 / w as f32).min(n_u32 as f32 / h as f32),
		FitMode::Cover => (n_u32 as f32 / w as f32).max(n_u32 as f32 / h as f32),
	};

	let new_w = ((w as f32 * scale).round().max(1.0)) as u32;
	let new_h = ((h as f32 * scale).round().max(1.0)) as u32;
	let resized = image::imageops::resize(&gray, new_w, new_h, FilterType::Lanczos3);

	let mut canvas = image::GrayImage::from_pixel(n_u32, n_u32, image::Luma([255u8]));

	let off_x = ((n_u32 as i64 - new_w as i64) / 2).max(0) as u32;
	let off_y = ((n_u32 as i64 - new_h as i64) / 2).max(0) as u32;

	if config.fit_mode == FitMode::Contain {
		image::imageops::replace(&mut canvas, &resized, off_x as i64, off_y as i64);
	} else {
		let crop_x = ((new_w as i64 - n_u32 as i64) / 2).max(0) as u32;
		let crop_y = ((new_h as i64 - n_u32 as i64) / 2).max(0) as u32;
		let cropped = image::imageops::crop_imm(&resized, crop_x, crop_y, n_u32, n_u32).to_image();
		image::imageops::replace(&mut canvas, &cropped, 0, 0);
	}

	let mut out = vec![vec![false; n]; n];
	for (y, row) in out.iter_mut().enumerate() {
		for (x, cell) in row.iter_mut().enumerate() {
			let lum = canvas.get_pixel(x as u32, y as u32).0[0];
			let mut dark = lum <= config.threshold;
			if config.invert {
				dark = !dark;
			}
			*cell = dark;
		}
	}
	Ok(out)
}

pub fn synthetic_circle_target(version: u8) -> Result<Matrix> {
	let n = module_size(version)?;
	let center = (n as f32 - 1.0) / 2.0;
	let radius = n as f32 * 0.36;
	let ring_inner = radius * 0.55;

	let mut out = vec![vec![false; n]; n];
	for (y, row) in out.iter_mut().enumerate() {
		for (x, cell) in row.iter_mut().enumerate() {
			let dx = x as f32 - center;
			let dy = y as f32 - center;
			let d = (dx * dx + dy * dy).sqrt();
			*cell = d <= radius && d >= ring_inner;
		}
	}
	Ok(out)
}
