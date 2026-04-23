use qrcodegen::{Mask, QrCode, QrCodeEcc, QrSegment, Version};

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

impl EccLevel {
	fn to_qrcodegen(self) -> QrCodeEcc {
		match self {
			Self::L => QrCodeEcc::Low,
			Self::M => QrCodeEcc::Medium,
			Self::Q => QrCodeEcc::Quartile,
			Self::H => QrCodeEcc::High,
		}
	}
}

impl MaskPattern {
	pub fn all() -> [MaskPattern; 8] {
		[
			Self::M0,
			Self::M1,
			Self::M2,
			Self::M3,
			Self::M4,
			Self::M5,
			Self::M6,
			Self::M7,
		]
	}

	pub fn from_u8(value: u8) -> Option<Self> {
		match value {
			0 => Some(Self::M0),
			1 => Some(Self::M1),
			2 => Some(Self::M2),
			3 => Some(Self::M3),
			4 => Some(Self::M4),
			5 => Some(Self::M5),
			6 => Some(Self::M6),
			7 => Some(Self::M7),
			_ => None,
		}
	}

	fn to_qrcodegen(self) -> Mask {
		Mask::new(match self {
			Self::M0 => 0,
			Self::M1 => 1,
			Self::M2 => 2,
			Self::M3 => 3,
			Self::M4 => 4,
			Self::M5 => 5,
			Self::M6 => 6,
			Self::M7 => 7,
		})
	}
}

fn version(spec: QrSpec) -> Result<Version> {
	if !(1..=40).contains(&spec.version) {
		return Err(QrodeError::UnsupportedVersion(spec.version));
	}
	Ok(Version::new(spec.version))
}

fn to_matrix(qr: &QrCode) -> Matrix {
	let size = qr.size();
	let mut modules = vec![vec![false; size as usize]; size as usize];
	for y in 0..size {
		for x in 0..size {
			modules[y as usize][x as usize] = qr.get_module(x, y);
		}
	}
	modules
}

fn encode_inner(spec: QrSpec, payload: &[u8], mask: Option<MaskPattern>) -> Result<EncodedQr> {
	let expected = capacity_bytes(spec)?;
	if payload.len() != expected {
		return Err(QrodeError::PayloadLengthMismatch {
			got: payload.len(),
			expected,
		});
	}

	let seg = QrSegment::make_bytes(payload);
	let ver = version(spec)?;
	let code = QrCode::encode_segments_advanced(
		&[seg],
		spec.ecc.to_qrcodegen(),
		ver,
		ver,
		mask.map(MaskPattern::to_qrcodegen),
		false,
	)
	.map_err(|e| QrodeError::Internal(format!("encode failed: {e:?}")))?;

	let actual_mask = match mask {
		Some(m) => m,
		None => MaskPattern::from_u8(code.mask().value() as u8).unwrap_or(MaskPattern::M0),
	};

	Ok(EncodedQr {
		spec,
		mask: actual_mask,
		modules: to_matrix(&code),
	})
}

pub fn module_size(version: u8) -> Result<usize> {
	if !(1..=40).contains(&version) {
		return Err(QrodeError::UnsupportedVersion(version));
	}
	Ok((21 + 4 * (version as usize - 1)) as usize)
}

pub fn capacity_bytes(spec: QrSpec) -> Result<usize> {
	if spec.mode != DataMode::Byte {
		return Err(QrodeError::Internal("only byte mode is supported".into()));
	}

	let mut lo = 0usize;
	let mut hi = 4096usize;
	while lo < hi {
		let mid = (lo + hi + 1) / 2;
		let payload = vec![0u8; mid];
		let seg = QrSegment::make_bytes(&payload);
		let ver = version(spec)?;
		let enc = QrCode::encode_segments_advanced(
			&[seg],
			spec.ecc.to_qrcodegen(),
			ver,
			ver,
			None,
			false,
		);
		if enc.is_ok() {
			lo = mid;
		} else {
			hi = mid - 1;
		}
	}
	Ok(lo)
}

pub fn encode_with_mask(spec: QrSpec, payload: &[u8], mask: MaskPattern) -> Result<EncodedQr> {
	encode_inner(spec, payload, Some(mask))
}

pub fn encode_auto_mask(spec: QrSpec, payload: &[u8]) -> Result<EncodedQr> {
	encode_inner(spec, payload, None)
}

pub fn immutable_mask(spec: QrSpec, mask: MaskPattern) -> Result<Matrix> {
	let cap = capacity_bytes(spec)?;
	let base_payload = vec![0u8; cap];
	let base = encode_with_mask(spec, &base_payload, mask)?;
	let size = base.modules.len();

	let mut mutable = vec![vec![false; size]; size];

	for bit_idx in 0..(cap * 8) {
		let mut payload = base_payload.clone();
		let byte_idx = bit_idx / 8;
		let bit = (bit_idx % 8) as u8;
		payload[byte_idx] ^= 1u8 << bit;
		let varied = encode_with_mask(spec, &payload, mask)?;

		for y in 0..size {
			for x in 0..size {
				if base.modules[y][x] != varied.modules[y][x] {
					mutable[y][x] = true;
				}
			}
		}
	}

	let mut immutable = vec![vec![false; size]; size];
	for y in 0..size {
		for x in 0..size {
			immutable[y][x] = !mutable[y][x];
		}
	}
	Ok(immutable)
}


