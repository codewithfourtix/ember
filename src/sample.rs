//! Turning a logit vector into the next token id.

use crate::ops::softmax;

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
    pub fn sample(&self, logits: &[f32], rng: &mut Rng) -> u32 {
        match *self {
            Sampler::Greedy => argmax(logits),
            Sampler::TopP { temperature, top_p } => {
                let mut probs: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
                softmax(&mut probs);

                // Sort token ids by probability, descending.
                let mut order: Vec<u32> = (0..probs.len() as u32).collect();
                order.sort_unstable_by(|&a, &b| {
                    probs[b as usize]
                        .partial_cmp(&probs[a as usize])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Keep the smallest prefix whose mass reaches `top_p`.
                let mut cumulative = 0.0f32;
                let mut cutoff = order.len();
                for (i, &id) in order.iter().enumerate() {
                    cumulative += probs[id as usize];
                    if cumulative >= top_p {
                        cutoff = i + 1;
                        break;
                    }
                }
                let nucleus = &order[..cutoff];

                // Sample from the renormalised nucleus.
                let mass: f32 = nucleus.iter().map(|&id| probs[id as usize]).sum();
                let mut r = rng.next_f32() * mass;
                for &id in nucleus {
                    r -= probs[id as usize];
                    if r <= 0.0 {
                        return id;
                    }
                }
                *nucleus.last().unwrap_or(&argmax(logits))
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

/// A tiny xorshift RNG — enough for sampling, and dependency-free.
pub struct Rng(u64);

impl Rng {
    /// Seed the generator (0 is remapped so the state is never all-zero).
    pub fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A float in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
