use crate::qr_core::{Matrix, QrodeError, Result};

#[derive(Clone, Copy, Debug)]
pub struct ScoreWeights {
    pub mutable_weight: f32,
    pub immutable_weight: f32,
    pub use_ssim: bool,
    pub ssim_weight: f32,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            mutable_weight: 1.0,
            immutable_weight: 0.05,
            use_ssim: true,
            ssim_weight: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ScoreBreakdown {
    pub objective_score: f32,
    pub ssim: f32,
    pub mutable_match_pct: f32,
    pub full_match_pct: f32,
    pub mutable_dark_error: f32,
    pub mutable_light_error: f32,
}

fn compute_ssim(candidate: &Matrix, target: &Matrix) -> f32 {
    let mut n = 0f32;
    let mut sum_x = 0f32;
    let mut sum_y = 0f32;
    let mut sum_x2 = 0f32;
    let mut sum_y2 = 0f32;
    let mut sum_xy = 0f32;

    for y in 0..candidate.len() {
        for x in 0..candidate[y].len() {
            let cx = if candidate[y][x] { 1.0f32 } else { 0.0f32 };
            let ty = if target[y][x] { 1.0f32 } else { 0.0f32 };
            n += 1.0;
            sum_x += cx;
            sum_y += ty;
            sum_x2 += cx * cx;
            sum_y2 += ty * ty;
            sum_xy += cx * ty;
        }
    }

    if n <= 1.0 {
        return 0.0;
    }

    let mu_x = sum_x / n;
    let mu_y = sum_y / n;
    let var_x = (sum_x2 / n) - mu_x * mu_x;
    let var_y = (sum_y2 / n) - mu_y * mu_y;
    let cov_xy = (sum_xy / n) - mu_x * mu_y;

    let c1 = 0.0001f32;
    let c2 = 0.0009f32;
    let numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2);
    let denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2);
    if denominator.abs() < f32::EPSILON {
        return 0.0;
    }

    (numerator / denominator).clamp(0.0, 1.0)
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
    let ssim = compute_ssim(candidate, target);

    let objective = if weights.use_ssim {
        weights.ssim_weight * ssim
    } else {
        weights.mutable_weight * mutable_match_ratio
            + weights.immutable_weight * immutable_match_ratio
    };

    Ok(ScoreBreakdown {
        objective_score: objective,
        ssim,
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
