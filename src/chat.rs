//! Qwen2.5 ChatML prompt formatting.
//!
//! The instruct models are trained on a specific chat layout; wrapping a user
//! turn in it (rather than feeding a raw string) is the difference between the
//! model *completing* your text and *answering* you. Generation then stops on
//! the `<|im_end|>` token that closes the assistant turn.

/// Wrap a user message in Qwen's ChatML template with the given system prompt.
pub fn chatml(system: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_user_and_system() {
        let p = chatml("You are helpful.", "Hi");
        assert!(p.starts_with("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(p.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }
}
