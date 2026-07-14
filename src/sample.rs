//! Turning a logit vector into the next token id.

/// Sampling strategy selected from the CLI.
#[derive(Debug, Clone, Copy)]
pub enum Sampler {
    /// Always take the arg-max (deterministic).
    Greedy,
    /// Temperature scaling followed by nucleus (top-p) sampling.
    TopP { temperature: f32, top_p: f32 },
}

impl Sampler {
    /// Pick the next token id from `logits`.
    pub fn sample(&self, logits: &[f32]) -> u32 {
        match *self {
            Sampler::Greedy => argmax(logits),
            Sampler::TopP { temperature, top_p } => {
                // Divide logits by `temperature`, softmax, keep the smallest set
                // of tokens whose probability mass ≥ `top_p`, renormalise, draw.
                let _ = (temperature, top_p);
                todo!("temperature + nucleus (top-p) sampling")
            }
        }
    }
}

/// Index of the maximum logit.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best as u32
}
