"""
LLM Interface
=============
Wraps Llama-3-8B-Instruct (or any HuggingFace causal LM) for generation.
The same frozen backbone is used across all retrieval methods to isolate
the effect of retrieval strategy (as described in Section IV-A).
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)


class LLMInterface:
    """
    HuggingFace-based LLM wrapper.

    Default: meta-llama/Meta-Llama-3-8B-Instruct
    Requires: pip install transformers accelerate torch
    Requires HuggingFace token for gated models (set HF_TOKEN env var).
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_new_tokens: int = 128,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        logger.info("Loading LLM: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map=device,
            max_new_tokens=max_new_tokens,
        )
        logger.info("LLM ready.")

    def generate(self, prompt: str) -> str:
        outputs = self.pipe(prompt, do_sample=False, temperature=None, top_p=None)
        generated = outputs[0]["generated_text"]
        # Strip the prompt prefix
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        return generated
