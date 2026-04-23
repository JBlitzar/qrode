use crate::qr_core::{Matrix, QrodeError, Result};

#[derive(Clone, Copy, Debug)]
pub struct ScoreWeights {
	pub mutable_weight: f32,
	pub immutable_weight: f32,
}

impl Default for ScoreWeights {
	fn default() -> Self {
		Self {
			mutable_weight: 1.0,
			immutable_weight: 0.05,
		}
	}
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ScoreBreakdown {
	pub objective_score: f32,
	pub mutable_match_pct: f32,
	pub full_match_pct: f32,
	pub mutable_dark_error: f32,
	pub mutable_light_error: f32,
}

pub fn score_modules(
	candidate: &Matrix,
	target: &Matrix,
	immutable_mask: &Matrix,
	weights: ScoreWeights,
) -> Result<ScoreBreakdown> {
	if candidate.len() != target.len() || candidate.len() != immutable_mask.len() {
		return Err(QrodeError::Internal("matrix height mismatch".into()));
	}

	let mut full_match = 0usize;
	let mut full_total = 0usize;

	let mut mutable_match = 0usize;
	let mut mutable_total = 0usize;

	let mut immutable_match = 0usize;
	let mut immutable_total = 0usize;

	let mut mutable_dark_error = 0usize;
	let mut mutable_light_error = 0usize;

	for y in 0..candidate.len() {
		if candidate[y].len() != target[y].len() || candidate[y].len() != immutable_mask[y].len() {
			return Err(QrodeError::Internal("matrix width mismatch".into()));
		}

		for x in 0..candidate[y].len() {
			let c = candidate[y][x];
			let t = target[y][x];
			let im = immutable_mask[y][x];

			full_total += 1;
			if c == t {
				full_match += 1;
			}

			if im {
				immutable_total += 1;
				if c == t {
					immutable_match += 1;
				}
			} else {
				mutable_total += 1;
				if c == t {
					mutable_match += 1;
				} else if t {
					mutable_dark_error += 1;
				} else {
					mutable_light_error += 1;
				}
			}
		}
	}

	let mutable_match_ratio = if mutable_total == 0 {
		0.0
	} else {
		mutable_match as f32 / mutable_total as f32
	};
	let immutable_match_ratio = if immutable_total == 0 {
		0.0
	} else {
		immutable_match as f32 / immutable_total as f32
	};
	let full_match_ratio = if full_total == 0 {
		0.0
	} else {
		full_match as f32 / full_total as f32
	};

	let objective = weights.mutable_weight * mutable_match_ratio
		+ weights.immutable_weight * immutable_match_ratio;

	Ok(ScoreBreakdown {
		objective_score: objective,
		mutable_match_pct: mutable_match_ratio * 100.0,
		full_match_pct: full_match_ratio * 100.0,
		mutable_dark_error: if mutable_total == 0 {
			0.0
		} else {
			mutable_dark_error as f32 / mutable_total as f32
		},
		mutable_light_error: if mutable_total == 0 {
			0.0
		} else {
			mutable_light_error as f32 / mutable_total as f32
		},
	})
}
