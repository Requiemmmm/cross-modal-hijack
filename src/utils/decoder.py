"""
Image decoder for converting VQ tokens to images.
"""

import os
from typing import List, Optional
import torch
from PIL import Image


class ImageDecoder:
    """Decode image tokens to PIL Image."""

    def __init__(self, model_loader):
        """
        Initialize decoder.

        Args:
            model_loader: ModelLoader instance with loaded model
        """
        self.loader = model_loader
        self.model = model_loader.model
        self.device = next(self.model.parameters()).device
        self.vqmodel = self.model.model.vqmodel

    def decode(
        self,
        image_tokens: List[int],
        expected_tokens: int = 1024,
    ) -> Optional[Image.Image]:
        """
        Decode image tokens to PIL Image.

        Args:
            image_tokens: List of image token IDs
            expected_tokens: Expected number of tokens (default 1024 for 32x32)

        Returns:
            PIL Image or None if decoding fails
        """
        if len(image_tokens) == 0:
            print("Warning: No image tokens to decode")
            return None

        # Pad or truncate to expected length
        if len(image_tokens) < expected_tokens:
            # Pad with first token (neutral fill)
            pad_token = image_tokens[0] if image_tokens else 4
            image_tokens = image_tokens + [pad_token] * (expected_tokens - len(image_tokens))
        elif len(image_tokens) > expected_tokens:
            image_tokens = image_tokens[:expected_tokens]

        # Convert to tensor and reshape to 32x32 grid
        token_tensor = torch.tensor(image_tokens, device=self.device)
        token_grid = token_tensor.view(1, 32, 32)

        try:
            with torch.no_grad():
                # Decode through VQ-GAN
                pixel_values = self.vqmodel.decode(token_grid)

                # Post-process to image
                pixel_values = pixel_values.clamp(-1, 1)
                pixel_values = (pixel_values + 1) / 2  # [-1, 1] -> [0, 1]
                pixel_values = pixel_values.squeeze(0).permute(1, 2, 0)  # CHW -> HWC
                pixel_values = (pixel_values * 255).cpu().numpy().astype('uint8')

                return Image.fromarray(pixel_values)

        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

    def decode_and_save(
        self,
        image_tokens: List[int],
        output_path: str,
        expected_tokens: int = 1024,
    ) -> bool:
        """
        Decode image tokens and save to file.

        Args:
            image_tokens: List of image token IDs
            output_path: Path to save the image
            expected_tokens: Expected number of tokens

        Returns:
            True if successful, False otherwise
        """
        image = self.decode(image_tokens, expected_tokens)

        if image is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            print(f"Image saved to: {output_path}")
            return True
        return False

    def decode_adversarial_patch(
        self,
        visual_token_ids: List[int],
        output_path: str,
    ) -> bool:
        """
        Decode adversarial visual tokens to a patch image.

        This creates a 512x512 image where the adversarial tokens
        occupy the first positions and the rest is neutral.

        Args:
            visual_token_ids: Adversarial visual token IDs
            output_path: Path to save the patch image

        Returns:
            True if successful, False otherwise
        """
        # Create 1024 token grid (32x32)
        # First N positions: adversarial tokens
        # Rest: neutral token (4 = black/neutral)
        neutral_token = 4
        full_tokens = visual_token_ids + [neutral_token] * (1024 - len(visual_token_ids))

        return self.decode_and_save(full_tokens, output_path, 1024)
