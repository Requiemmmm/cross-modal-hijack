"""
Hybrid Prefix Attack - The successful attack method.

This attack combines:
1. System-level instruction (<SYSTEM> tags) for priority attention
2. Visual adversarial tokens (32 tokens at phase transition threshold)
3. Prefix positioning (instruction before victim prompt)

Key insight: The three factors have multiplicative effect -
System instruction alone or more tokens alone may fail,
but together they achieve successful cross-modal hijacking.
"""

import os
from typing import List, Tuple, Dict, Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttack
from .hardening import GradientGuidedHardening


class HybridPrefixAttack(BaseAttack):
    """
    Hybrid Prefix Attack for cross-modal hijacking.

    This is the successful attack method that generates dog images
    when the victim prompt asks for cat images.

    Attack structure:
    [SYSTEM instruction] [BOI] [32 visual tokens] [EOI] [victim prompt]
    """

    def __init__(
        self,
        model_loader,
        prefix_length: int = 32,
        instruction_style: str = "system",
    ):
        super().__init__(model_loader, prefix_length, instruction_style)
        self.hardener = GradientGuidedHardening(self)

    def init_soft_prefix(
        self,
        target_image_tokens: List[int],
        noise_scale: float = 0.02,
    ) -> nn.Parameter:
        """
        Initialize visual soft prefix from target image tokens.

        Strategy: Start from target dog image tokens with small noise.
        This provides a good initialization point close to the solution.
        """
        # Take first prefix_length tokens from target
        sample_tokens = target_image_tokens[:self.prefix_length]
        sample_ids = torch.tensor([sample_tokens], device=self.device, dtype=torch.long)

        with torch.no_grad():
            sample_emb = self.embedding(sample_ids)
            noise = torch.randn_like(sample_emb) * noise_scale
            base = sample_emb + noise

        param = nn.Parameter(base.float())
        param.requires_grad = True
        return param

    def compute_loss(
        self,
        visual_soft_prefix: torch.Tensor,
        victim_prompt: str,
        target_image_tokens: List[int],
        instruction_text: Optional[str] = None,
        match_image_len: int = 128,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute hybrid prefix loss.

        Prompt structure for teacher forcing:
        [instruction] [BOI] [visual_prefix] [EOI] [victim] [BOI] [target_image] [EOI]

        Loss: Cross-entropy on predicting target image tokens.
        """
        if instruction_text is None:
            instruction_text = self.get_instruction_text()

        # Encode components
        instruction_ids = self.encode_text(instruction_text)
        victim_ids = self.encode_text(victim_prompt)

        boi = torch.tensor([[self.boi_id]], device=self.device, dtype=torch.long)
        eoi = torch.tensor([[self.eoi_id]], device=self.device, dtype=torch.long)

        target_img_ids = torch.tensor(
            [target_image_tokens[:match_image_len]],
            device=self.device,
            dtype=torch.long
        )

        # Get embeddings
        instruction_embeds = self.embedding(instruction_ids)
        victim_embeds = self.embedding(victim_ids)
        boi_embeds = self.embedding(boi)
        eoi_embeds = self.embedding(eoi)

        with torch.no_grad():
            target_img_embeds = self.embedding(target_img_ids)

        # Build input sequence - INSTRUCTION FIRST (key for success)
        input_embeds = torch.cat([
            instruction_embeds,                          # System instruction
            boi_embeds,                                  # [BOI]
            visual_soft_prefix.to(self.model.dtype),     # Visual prefix
            eoi_embeds,                                  # [EOI]
            victim_embeds,                               # "a photo of a cat"
            boi_embeds,                                  # [BOI]
            target_img_embeds,                           # Target dog image
            eoi_embeds                                   # [EOI]
        ], dim=1)

        # Forward pass
        outputs = self.model.model(
            inputs_embeds=input_embeds,
            use_cache=False,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # Compute context length
        instruction_len = instruction_ids.shape[1]
        prefix_len = visual_soft_prefix.shape[1]
        victim_len = victim_ids.shape[1]
        context_len = instruction_len + 1 + prefix_len + 1 + victim_len + 1

        # Image generation loss
        image_logits = logits[:, context_len-1:context_len-1+match_image_len, :]
        image_targets = target_img_ids[:, :match_image_len]

        min_len = min(image_logits.shape[1], image_targets.shape[1])
        loss = F.cross_entropy(
            image_logits[:, :min_len, :].reshape(-1, self.vocab_size),
            image_targets[:, :min_len].reshape(-1),
            reduction='mean'
        )

        loss_dict = {
            'total': loss.item(),
            'image': loss.item(),
        }

        return loss, loss_dict

    def optimize(
        self,
        victim_prompt: str,
        target_image_tokens: List[int],
        num_epochs: int = 150,
        learning_rate: float = 0.01,
        match_image_len: int = 128,
        output_dir: Optional[str] = None,
    ) -> Tuple[nn.Parameter, Dict]:
        """
        Optimize visual soft prefix.

        Args:
            victim_prompt: The prompt to hijack (e.g., "a photo of a cat")
            target_image_tokens: Target image tokens (e.g., dog image)
            num_epochs: Number of optimization epochs
            learning_rate: Learning rate for AdamW
            match_image_len: Number of image tokens to match
            output_dir: Directory to save results

        Returns:
            Tuple of (optimized soft prefix, results dict)
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        instruction_text = self.get_instruction_text()

        print(f"\n{'='*70}")
        print("Hybrid Prefix Attack - Soft Optimization")
        print(f"{'='*70}")
        print(f"Instruction: '{instruction_text}'")
        print(f"Visual prefix length: {self.prefix_length}")
        print(f"Victim prompt: '{victim_prompt}'")
        print(f"{'='*70}\n")

        # Initialize soft prefix
        visual_soft = self.init_soft_prefix(target_image_tokens)
        print(f"Soft prefix shape: {visual_soft.shape}")

        # Setup optimizer
        optimizer = torch.optim.AdamW([visual_soft], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        best_loss = float('inf')
        best_prefix = None
        history = []

        print("\nStarting optimization...")
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()

            loss, loss_dict = self.compute_loss(
                visual_soft,
                victim_prompt,
                target_image_tokens,
                instruction_text,
                match_image_len,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([visual_soft], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            history.append(loss_dict)

            if loss_dict['total'] < best_loss:
                best_loss = loss_dict['total']
                best_prefix = visual_soft.detach().clone()

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d} | Loss: {loss_dict['total']:.6f}")

            if epoch % 50 == 0:
                torch.cuda.empty_cache()

        final_prefix = best_prefix if best_prefix is not None else visual_soft.detach()

        # Save checkpoint
        if output_dir:
            checkpoint = {
                'visual_soft_prefix': final_prefix.cpu(),
                'instruction_text': instruction_text,
                'best_loss': best_loss,
                'history': history,
                'config': {
                    'instruction_text': instruction_text,
                    'victim_prompt': victim_prompt,
                    'prefix_length': self.prefix_length,
                    'instruction_style': self.instruction_style,
                    'match_image_len': match_image_len,
                    'num_epochs': num_epochs,
                }
            }

            save_path = os.path.join(output_dir, "soft_prefix.pt")
            torch.save(checkpoint, save_path)
            print(f"\nSaved soft prefix to: {save_path}")

        results = {
            'best_loss': best_loss,
            'history': history,
            'instruction_text': instruction_text,
        }

        return final_prefix, results

    def harden(
        self,
        soft_prefix: torch.Tensor,
        victim_prompt: str,
        target_image_tokens: List[int],
        instruction_text: Optional[str] = None,
        match_image_len: int = 128,
        top_k: int = 64,
        batch_size: int = 32,
        output_dir: Optional[str] = None,
    ) -> List[int]:
        """
        Convert soft prefix to discrete tokens using gradient-guided hardening.

        This is the critical step that bridges white-box (soft) to black-box (discrete).
        """
        if instruction_text is None:
            instruction_text = self.get_instruction_text()

        discrete_ids = self.hardener.harden(
            soft_prefix.to(self.device),
            victim_prompt,
            target_image_tokens,
            instruction_text,
            match_image_len,
            top_k,
            batch_size,
        )

        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results = {
                'discrete_prefix_ids': discrete_ids,
                'instruction_text': instruction_text,
                'instruction_style': self.instruction_style,
            }
            save_path = os.path.join(output_dir, "discrete_prefix.json")
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved discrete prefix to: {save_path}")

        return discrete_ids

    @torch.no_grad()
    def verify(
        self,
        discrete_prefix: List[int],
        victim_prompt: str,
        instruction_text: Optional[str] = None,
        max_new_tokens: int = 1200,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Verify attack by generating with discrete prefix.

        Returns:
            Dict with generated text, image tokens, and full output.
        """
        if instruction_text is None:
            instruction_text = self.get_instruction_text()

        print("\n" + "="*70)
        print("Verifying Hybrid Prefix Attack")
        print("="*70)
        print(f"Prompt structure:")
        print(f"  '{instruction_text}' + [BOI][{len(discrete_prefix)} tokens][EOI] + '{victim_prompt}'")
        print("="*70)

        # Build input
        instruction_ids = self.encode_text(instruction_text)
        victim_ids = self.encode_text(victim_prompt)
        boi = torch.tensor([[self.boi_id]], device=self.device, dtype=torch.long)
        eoi = torch.tensor([[self.eoi_id]], device=self.device, dtype=torch.long)
        visual_ids = torch.tensor([discrete_prefix], device=self.device, dtype=torch.long)

        input_ids = torch.cat([instruction_ids, boi, visual_ids, eoi, victim_ids], dim=1)
        current_embeds = self.embedding(input_ids).to(self.model.dtype)

        generated_ids = []

        # Autoregressive generation
        for step in range(max_new_tokens):
            outputs = self.model.model(
                inputs_embeds=current_embeds,
                use_cache=False,
            )

            logits = self.lm_head(outputs.last_hidden_state[:, -1:, :])
            next_id = torch.argmax(logits, dim=-1).item()
            generated_ids.append(next_id)

            if next_id == self.eoi_id:
                break

            next_emb = self.embedding(torch.tensor([[next_id]], device=self.device))
            current_embeds = torch.cat([current_embeds, next_emb.to(self.model.dtype)], dim=1)

        # Parse output
        text_tokens = []
        image_tokens = []
        in_image = False

        for tid in generated_ids:
            if tid == self.boi_id:
                in_image = True
            elif tid == self.eoi_id:
                in_image = False
            elif in_image:
                image_tokens.append(tid)
            else:
                text_tokens.append(tid)

        text_output = self.loader.decode_tokens(text_tokens)

        print(f"\nGenerated text: '{text_output[:200]}...'")
        print(f"Generated {len(image_tokens)} image tokens")
        print("="*70)

        results = {
            'instruction_text': instruction_text,
            'discrete_prefix_ids': discrete_prefix,
            'generated_text': text_output,
            'full_tokens': generated_ids,
            'image_tokens': image_tokens,
            'num_image_tokens': len(image_tokens),
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "verification_results.json")
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved verification to: {save_path}")

        return results
