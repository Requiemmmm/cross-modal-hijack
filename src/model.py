"""
Model loading utilities for Chameleon/Anole-7B.
"""

import os
import sys
from typing import Optional

import torch


def setup_environment(repo_root: Optional[str] = None):
    """Setup Python path to use local transformers."""
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    local_tf_src = os.path.join(repo_root, "transformers", "src")
    if os.path.isdir(local_tf_src) and local_tf_src not in sys.path:
        sys.path.insert(0, local_tf_src)

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# Setup environment on import
setup_environment()

from transformers import (
    ChameleonConfig,
    ChameleonProcessor,
    ChameleonForConditionalGenerationWithCFG,
)


class ModelLoader:
    """Utility class for loading Chameleon/Anole-7B model."""

    def __init__(
        self,
        model_path: str = "twgi-critique-anole-7b",
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.config = None

    def load(self, freeze: bool = True, gradient_checkpointing: bool = True):
        """Load model and processor."""
        print(f"Loading model from {self.model_path}...")

        # Load config
        self.config = ChameleonConfig.from_pretrained(self.model_path)
        self.config._attn_implementation = "eager"

        # Load processor and tokenizer
        self.processor = ChameleonProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer

        # Load model
        self.model = ChameleonForConditionalGenerationWithCFG.from_pretrained(
            self.model_path,
            config=self.config,
            torch_dtype=self.dtype,
        ).to(self.device)

        # Freeze parameters if requested
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        # Setup for training mode (needed for gradient computation)
        self.model.train()

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Disable CFG
        self.model.setup_cfg(cfg_config="no")

        print(f"Model loaded successfully.")
        print(f"  - Hidden size: {self.config.hidden_size}")
        print(f"  - Vocab size: {self.config.vocab_size}")
        print(f"  - BOI token: {self.config.boi_token_id}")
        print(f"  - EOI token: {self.config.eoi_token_id}")

        return self

    @property
    def embedding(self):
        """Get embedding layer."""
        return self.model.model.embed_tokens

    @property
    def lm_head(self):
        """Get language model head."""
        return self.model.lm_head

    @property
    def boi_id(self):
        """Get BOI (Begin of Image) token ID."""
        return self.config.boi_token_id

    @property
    def eoi_id(self):
        """Get EOI (End of Image) token ID."""
        return self.config.eoi_token_id

    @property
    def image_token_ids(self):
        """Get list of image token IDs."""
        return list(self.model.model.vocabulary_mapping.image_token_ids)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    def decode_tokens(self, token_ids: list) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
