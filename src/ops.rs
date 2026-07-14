//! Element-wise transformer building blocks: normalisation, positional
//! encoding, activation, and softmax. Each is a small, self-contained kernel.

/// RMSNorm: `out = x / sqrt(mean(x²) + ε) * weight`.
pub fn rms_norm(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());
    let n = x.len() as f32;
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let scale = 1.0 / (ss / n + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

/// Apply rotary position embeddings (RoPE) in place to a single head's query or
/// key vector at absolute position `pos`.
///
/// This uses the HuggingFace "rotate-half" convention: dimension `j` pairs with
/// `j + head_dim/2` (not the adjacent index), which is what Llama/Qwen weights
/// expect. Matching this exactly is the difference between coherent text and
/// noise.
pub fn rope(vec: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    debug_assert_eq!(vec.len(), head_dim);
    let half = head_dim / 2;
    for j in 0..half {
        let freq = 1.0 / theta.powf(2.0 * j as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let x0 = vec[j];
        let x1 = vec[j + half];
        vec[j] = x0 * cos - x1 * sin;
        vec[j + half] = x1 * cos + x0 * sin;
    }
}

/// SwiGLU feed-forward activation: `out = silu(gate) ⊙ up`, where
/// `silu(z) = z · σ(z)`.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(out.len(), up.len());
    for i in 0..gate.len() {
        let z = gate[i];
        let silu = z / (1.0 + (-z).exp());
        out[i] = silu * up[i];
    }
}

/// Numerically-stable softmax over `x`, in place (subtract max, exp, normalise).
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }

    #[test]
    fn softmax_uniform() {
        let mut x = [0.0, 0.0];
        softmax(&mut x);
        approx(x[0], 0.5);
        approx(x[1], 0.5);
    }

    #[test]
    fn softmax_normalises_and_orders() {
        let mut x = [1.0, 2.0, 3.0, -1.0];
        softmax(&mut x);
        approx(x.iter().sum::<f32>(), 1.0);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn swiglu_matches_silu() {
        // silu(1)·2 = (1·σ(1))·2 = 0.7310586·2 = 1.4621172
        let mut out = [0.0];
        swiglu(&[1.0], &[2.0], &mut out);
        approx(out[0], 1.4621172);
    }

    #[test]
    fn rms_norm_gives_unit_rms() {
        // With unit weights, the output RMS should be ~1.
        let x = [3.0, 4.0, -5.0, 2.0];
        let w = [1.0; 4];
        let mut out = [0.0; 4];
        rms_norm(&x, &w, &mut out, 1e-6);
        let mean_sq: f32 = out.iter().map(|v| v * v).sum::<f32>() / 4.0;
        approx(mean_sq, 1.0);
    }

    #[test]
    fn rope_preserves_norm() {
        // A rotation must not change a vector's length.
        let mut v = [0.5, -1.0, 2.0, 0.3, 1.0, 0.0, -0.7, 0.9];
        let before: f32 = v.iter().map(|x| x * x).sum();
        rope(&mut v, 5, 8, 10_000.0);
        let after: f32 = v.iter().map(|x| x * x).sum();
        approx(before, after);
    }
}
