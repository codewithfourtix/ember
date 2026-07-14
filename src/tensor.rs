//! The numeric core: dense `f32` linear algebra.
//!
//! A decode step is dominated by matrix–vector products against the weight
//! matrices, so [`matvec`] is the single hottest routine in the engine. It is
//! written by hand (and parallelised with `rayon`) rather than pulled from a
//! BLAS / `ndarray` crate — implementing it is the point.

use rayon::prelude::*;

/// Row-major matrix–vector product `y = W · x`.
///
/// `w` is `[out_dim × in_dim]` in row-major order, `x` is `[in_dim]`, and the
/// `[out_dim]` result is written into `y`. Each output row is an independent dot
/// product, so the rows are computed in parallel.
pub fn matvec(w: &[f32], x: &[f32], y: &mut [f32], in_dim: usize, out_dim: usize) {
    debug_assert_eq!(w.len(), in_dim * out_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);

    y.par_iter_mut().enumerate().for_each(|(o, yo)| {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            acc += row[i] * x[i];
        }
        *yo = acc;
    });
}

/// In-place element-wise add: `a += b` (transformer residual connections).
pub fn add_assign(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (ai, bi) in a.iter_mut().zip(b) {
        *ai += *bi;
    }
}

/// In-place element-wise add of a bias vector: `a += bias`.
pub fn add_bias(a: &mut [f32], bias: &[f32]) {
    add_assign(a, bias);
}
