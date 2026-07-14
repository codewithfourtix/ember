//! Weight quantization — the headline Phase 2 optimization.
//!
//! Weights dominate both the memory footprint and the memory *bandwidth* that
//! bottlenecks CPU decoding. Storing them as INT8/INT4 with per-block scales
//! shrinks the model and, because decode is bandwidth-bound, speeds it up. The
//! forward path dequantizes on the fly inside the mat-vec.
//!
//! Two schemes, both verified to keep Qwen2.5-0.5B coherent:
//!   * **INT8**, one scale per row — ~4× smaller, output near-identical to f32.
//!   * **INT4**, one scale per group of [`INT4_GROUP`] columns (packed two per
//!     byte) — ~7× smaller and still coherent. Per-*row* INT4 is too coarse and
//!     degrades badly, which is exactly why grouping matters.

use rayon::prelude::*;

use crate::tensor::matvec;

/// Number of columns that share one INT4 scale. Divides every matrix dimension
/// in the Qwen2.5 models (896, 128, 4864 are all multiples of 64).
pub const INT4_GROUP: usize = 64;

/// Quantization scheme selected from the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    None,
    Int8,
    Int4,
}

impl Quant {
    pub fn parse(s: &str) -> Option<Quant> {
        match s.to_ascii_lowercase().as_str() {
            "none" | "f32" => Some(Quant::None),
            "int8" | "q8" => Some(Quant::Int8),
            "int4" | "q4" => Some(Quant::Int4),
            _ => None,
        }
    }
}

/// A quantized weight matrix: packed integers plus per-block scales.
pub struct QuantMatrix {
    /// INT8: one signed byte per weight. INT4: two 4-bit weights per byte.
    packed: Vec<u8>,
    /// One dequantization scale per (row, group).
    scales: Vec<f32>,
    rows: usize,
    cols: usize,
    bits: u8,
    group: usize,
}

impl QuantMatrix {
    /// Symmetric block quantization of a row-major `[rows × cols]` matrix.
    fn quantize(w: &[f32], rows: usize, cols: usize, bits: u8, group: usize) -> Self {
        debug_assert_eq!(w.len(), rows * cols);
        debug_assert_eq!(cols % group, 0);
        let qmax = ((1i32 << (bits - 1)) - 1) as f32; // 127 (int8) or 7 (int4)
        let n_groups = cols / group;
        let mut scales = vec![0.0f32; rows * n_groups];
        let mut packed = vec![0u8; if bits == 8 { rows * cols } else { rows * cols / 2 }];

        for r in 0..rows {
            for g in 0..n_groups {
                // Per-group scale from the max magnitude.
                let mut amax = 0.0f32;
                for c in 0..group {
                    amax = amax.max(w[r * cols + g * group + c].abs());
                }
                let scale = if amax == 0.0 { 1.0 } else { amax / qmax };
                scales[r * n_groups + g] = scale;

                for c in 0..group {
                    let idx = r * cols + g * group + c;
                    let q = (w[idx] / scale).round().clamp(-qmax, qmax) as i32;
                    if bits == 8 {
                        packed[idx] = (q as i8) as u8;
                    } else {
                        let nib = (q & 0x0F) as u8;
                        let p = idx / 2;
                        if idx % 2 == 0 {
                            packed[p] = (packed[p] & 0xF0) | nib;
                        } else {
                            packed[p] = (packed[p] & 0x0F) | (nib << 4);
                        }
                    }
                }
            }
        }
        Self { packed, scales, rows, cols, bits, group }
    }

    /// The signed integer weight at `(row, col)`.
    #[inline]
    fn weight(&self, row: usize, col: usize) -> i32 {
        let idx = row * self.cols + col;
        if self.bits == 8 {
            (self.packed[idx] as i8) as i32
        } else {
            let byte = self.packed[idx / 2];
            let nib = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 } as i32;
            if nib >= 8 { nib - 16 } else { nib } // sign-extend 4 bits
        }
    }

    /// Fused dequantize + mat-vec: `y = dequant(self) · x`.
    fn matvec(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(y.len(), self.rows);
        let n_groups = self.cols / self.group;
        y.par_iter_mut().enumerate().for_each(|(r, yr)| {
            let mut acc = 0.0f32;
            for g in 0..n_groups {
                let scale = self.scales[r * n_groups + g];
                let mut partial = 0.0f32;
                for c in 0..self.group {
                    let col = g * self.group + c;
                    partial += self.weight(r, col) as f32 * x[col];
                }
                acc += partial * scale;
            }
            *yr = acc;
        });
    }

    /// Dequantize a single row (used for the embedding lookup).
    fn dequant_row(&self, row: usize) -> Vec<f32> {
        let n_groups = self.cols / self.group;
        let mut out = vec![0.0f32; self.cols];
        for g in 0..n_groups {
            let scale = self.scales[row * n_groups + g];
            for c in 0..self.group {
                let col = g * self.group + c;
                out[col] = self.weight(row, col) as f32 * scale;
            }
        }
        out
    }

    fn bytes(&self) -> usize {
        self.packed.len() + self.scales.len() * 4
    }
}

/// A linear weight that is either dense `f32` or quantized — so the model can
/// hold both without the forward pass caring which.
pub enum Linear {
    Dense { w: Vec<f32>, rows: usize, cols: usize },
    Quantized(QuantMatrix),
}

impl Linear {
    /// Build from a dense `f32` matrix, quantizing it per `scheme`.
    pub fn build(w: Vec<f32>, rows: usize, cols: usize, scheme: Quant) -> Self {
        match scheme {
            Quant::None => Linear::Dense { w, rows, cols },
            Quant::Int8 => Linear::Quantized(QuantMatrix::quantize(&w, rows, cols, 8, cols)),
            Quant::Int4 => Linear::Quantized(QuantMatrix::quantize(&w, rows, cols, 4, INT4_GROUP)),
        }
    }

    /// `y = W · x`.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        match self {
            Linear::Dense { w, rows, cols } => matvec(w, x, y, *cols, *rows),
            Linear::Quantized(q) => q.matvec(x, y),
        }
    }

    /// Row `r` of the matrix as `f32` (for the embedding lookup).
    pub fn row(&self, r: usize) -> Vec<f32> {
        match self {
            Linear::Dense { w, cols, .. } => w[r * cols..(r + 1) * cols].to_vec(),
            Linear::Quantized(q) => q.dequant_row(r),
        }
    }

    /// Approximate in-memory size in bytes (for the benchmark report).
    pub fn bytes(&self) -> usize {
        match self {
            Linear::Dense { w, .. } => w.len() * 4,
            Linear::Quantized(q) => q.bytes(),
        }
    }
}
