//! Weight quantization — the headline optimization.
//!
//! Weights dominate both the memory footprint and the memory *bandwidth* that
//! bottlenecks CPU decoding. Storing them as INT8/INT4 with a per-row scale
//! shrinks the model ~2×/~4× and, because decode is bandwidth-bound, speeds it
//! up too. The forward path dequantizes on the fly inside the mat-vec, so the
//! weights are never materialised back to `f32`.

/// Quantization width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBits {
    /// 8-bit signed, one value per byte.
    Int8,
    /// 4-bit signed, two values packed per byte.
    Int4,
}

/// A weight matrix stored as row-wise-quantized integers plus per-row scales.
pub struct QuantMatrix {
    /// Packed quantized weights (`i8`, or two 4-bit values per byte for INT4).
    pub packed: Vec<i8>,
    /// One dequantization scale per output row.
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub bits: QuantBits,
}

/// Quantize a row-major `[rows × cols]` `f32` matrix with one symmetric scale
/// per row: `scale = max|w_row| / qmax`, `q = round(w / scale)`.
pub fn quantize_rowwise(w: &[f32], rows: usize, cols: usize, bits: QuantBits) -> QuantMatrix {
    debug_assert_eq!(w.len(), rows * cols);
    let _ = (w, rows, cols, bits);
    todo!("row-wise symmetric quantization")
}

/// Fused dequantize + mat-vec: `y = dequant(q) · x`, without ever expanding the
/// weights to `f32`.
pub fn matvec_quant(q: &QuantMatrix, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), q.cols);
    debug_assert_eq!(y.len(), q.rows);
    let _ = (q, x, y);
    todo!("fused dequantize + mat-vec")
}
