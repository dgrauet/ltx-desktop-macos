"""Prompt enhancement via Qwen3.5-2B (lazy load/unload).

Loads the model only for the duration of a single enhance() call,
then immediately unloads and calls aggressive_cleanup(). The video model
and the prompt enhancer must NEVER coexist in memory on machines < 64GB.
"""

from __future__ import annotations

import logging

from engine.memory_manager import aggressive_cleanup

log = logging.getLogger(__name__)

# System prompt that rewrites short prompts into detailed LTX-2.3 optimised prompts.
_SYSTEM_PROMPT = (
    "You are a prompt enhancer for AI video generation with LTX-2.3. "
    "Rewrite the user's short description into a single detailed, cinematic paragraph. "
    "Include: (1) main subject — appearance, clothing, expressions; "
    "(2) specific, chronological actions and movements; "
    "(3) environment — location, lighting, atmosphere; "
    "(4) camera — angle and movement (pan, dolly, static, tracking); "
    "(5) visual style — cinematic, realistic, animation, colour grading; "
    "(6) audio elements — ambient sounds, music, any dialogue in quotes. "
    "Start directly with the action. Do not use bullet points or headers. "
    "Keep the response under 200 words. Output ONLY the enhanced prompt, nothing else."
)


class PromptEnhancer:
    """Enhances short prompts into detailed LTX-2.3 optimised prompts via Qwen3.5-2B.

    The model is loaded immediately before generation and unloaded immediately
    after. It is never kept resident between calls.
    """

    MODEL_ID: str = "mlx-community/Qwen3.5-2B-4bit"

    @classmethod
    def is_available(cls) -> bool:
        """Return True if mlx-lm is importable and enhancement is possible.

        Returns:
            True when mlx_lm can be imported, False otherwise.
        """
        try:
            import mlx_lm  # noqa: F401

            return True
        except ImportError:
            return False

    def enhance(self, prompt: str) -> str:
        """Load Qwen3.5-2B, enhance the prompt, then unload immediately.

        Args:
            prompt: Short user-supplied video description.

        Returns:
            Detailed, LTX-2.3-optimised prompt string.

        Raises:
            RuntimeError: If mlx-lm is not installed.
        """
        if not self.is_available():
            raise RuntimeError(
                "mlx-lm not installed. Run: uv add mlx-lm"
            )

        # Import here so that the module-level import does not fail when
        # mlx-lm is absent — the classmethod is_available() guards this path.
        from mlx_lm import generate, load  # type: ignore[import-untyped]

        log.info("PromptEnhancer: loading %s", self.MODEL_ID)
        model, tokenizer = load(self.MODEL_ID)
        log.info("PromptEnhancer: model loaded — enhancing prompt")

        # Build the chat-formatted prompt expected by Qwen instruction models.
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        formatted: str
        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback for tokenizers that don't support apply_chat_template.
            formatted = f"System: {_SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"

        enhanced: str = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=300,
            verbose=False,
        )

        log.info("PromptEnhancer: enhancement complete — unloading model")
        del model, tokenizer
        aggressive_cleanup()
        log.info("PromptEnhancer: model unloaded and memory freed")

        return enhanced.strip()
