use std::path::Path;

use crate::qr_core::{Matrix, QrodeError, Result};

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
) -> Result<ValidationReport> {
    let length_ok = payload.len() == expected_len;
    let prefix_ok = payload.starts_with(prefix);
    let ok = length_ok && prefix_ok;

    Ok(ValidationReport {
        prefix_ok,
        length_ok,
        decode_ok: ok,
        decoded_payload: None,
        message: if ok {
            "payload policy valid".into()
        } else {
            "payload policy invalid".into()
        },
    })
}

pub fn validate_with_decoder(png_path: &Path, expected_payload: &[u8]) -> Result<ValidationReport> {
    let img = image::open(png_path)
        .map_err(|e| QrodeError::Validation(format!("failed to open png: {e}")))?
        .to_luma8();
    let (w, h) = img.dimensions();

    let mut decoder = quircs::Quirc::default();
    let mut identified = decoder.identify(w as usize, h as usize, &img);
    while let Some(code) = identified.next() {
        let code = code.map_err(|e| QrodeError::Validation(format!("identify error: {e:?}")))?;
        let decoded = code
            .decode()
            .map_err(|e| QrodeError::Validation(format!("decode error: {e:?}")))?;

        let payload = decoded.payload;
        let decode_ok = payload == expected_payload;
        return Ok(ValidationReport {
            prefix_ok: true,
            length_ok: payload.len() == expected_payload.len(),
            decode_ok,
            decoded_payload: Some(payload),
            message: if decode_ok {
                "decoder payload matched".into()
            } else {
                "decoder payload mismatch".into()
            },
        });
    }

    Err(QrodeError::Validation("no qr code found by decoder".into()))
}

pub fn decode_modules_payload(modules: &Matrix) -> Option<Vec<u8>> {
    if modules.is_empty() || modules[0].is_empty() {
        return None;
    }

    let module_px = 4usize;
    let quiet_zone = 4usize;
    let n = modules.len();
    let size = (n + 2 * quiet_zone) * module_px;
    let mut gray = vec![255u8; size * size];

    for (y, row) in modules.iter().enumerate() {
        for (x, &dark) in row.iter().enumerate() {
            let value = if dark { 0u8 } else { 255u8 };
            let sx = (x + quiet_zone) * module_px;
            let sy = (y + quiet_zone) * module_px;
            for py in sy..(sy + module_px) {
                let row_start = py * size;
                for px in sx..(sx + module_px) {
                    gray[row_start + px] = value;
                }
            }
        }
    }

    let mut decoder = quircs::Quirc::default();
    let mut identified = decoder.identify(size, size, &gray);
    while let Some(code) = identified.next() {
        let code = code.ok()?;
        let decoded = code.decode().ok()?;
        return Some(decoded.payload);
    }

    None
}
