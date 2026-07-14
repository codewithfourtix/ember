//! Element-wise transformer building blocks: normalisation, positional
//! encoding, activation, and softmax. Each is a small, self-contained kernel.

/// RMSNorm: `out = x / sqrt(mean(x²) + ε) * weight`.
pub fn rms_norm(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());
    let _ = (x, weight, out, eps);
    todo!("RMSNorm: normalise by the RMS of x, then scale by `weight`")
}

/// Apply rotary position embeddings (RoPE) in place to a single head's query or
/// key vector at absolute position `pos`.
pub fn rope(vec: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    debug_assert_eq!(vec.len() % head_dim, 0);
    // For each (even, odd) dimension pair `i`, rotate by angle
    //   pos / theta^(2i / head_dim).
    let _ = (vec, pos, head_dim, theta);
    todo!("RoPE rotation of (even, odd) dimension pairs")
}

/// SwiGLU feed-forward activation: `out = silu(gate) ⊙ up`, where
/// `silu(z) = z * σ(z)`.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(out.len(), up.len());
    let _ = (gate, up, out);
    todo!("SwiGLU: silu(gate) elementwise-times up")
}

/// Numerically-stable softmax over `x`, in place (subtract max, exp, normalise).
pub fn softmax(x: &mut [f32]) {
    let _ = x;
    todo!("stable softmax over the attention scores")
}
