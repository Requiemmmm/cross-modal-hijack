"""
Gradient-Guided Hardening for converting soft prompts to discrete tokens.

Key insight from v3.1: Each position must use FRESH gradients computed
after previous positions are already discretized, because the model
is autoregressive and earlier choices affect later optimal solutions.
"""

from typing import List, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .base import BaseAttack


class GradientGuidedHardening:
    """
    Convert soft embeddings to discrete tokens using gradient guidance.

    This implements the HotFlip-inspired approach with iterative
    gradient updates to avoid stale gradient issues.
    """

    def __init__(self, attack: "BaseAttack"):
        self.attack = attack
        self.device = attack.device
        self.embedding = attack.embedding
        self.image_token_ids = attack.image_token_ids

    def harden(
        self,
        soft_prefix: torch.Tensor,
        victim_prompt: str,
        target_image_tokens: List[int],
        instruction_text: str,
        match_image_len: int = 128,
        top_k: int = 64,
        batch_size: int = 32,
    ) -> List[int]:
        """
        Convert soft prefix to discrete tokens.

        Algorithm:
        1. For each position i:
           a. Compute FRESH gradient (not reusing old gradients)
           b. Use HotFlip to select top-k candidates from image vocabulary
           c. Evaluate each candidate's actual loss
           d. Select best candidate
           e. Update current embedding for next position

        Args:
            soft_prefix: Soft embedding tensor [1, prefix_len, hidden_size]
            victim_prompt: Victim prompt text
            target_image_tokens: Target image token IDs
            instruction_text: Instruction text
            match_image_len: Number of image tokens to match
            top_k: Number of HotFlip candidates
            batch_size: Number of candidates to evaluate

        Returns:
            List of discrete token IDs
        """
        discrete_ids = []
        embed_weight = self.embedding.weight
        prefix_len = soft_prefix.shape[1]

        print(f"\n    Gradient-Guided Hardening ({prefix_len} positions)...")
        print(f"    Using fresh gradients per position (v3.1 fix)")

        # Start with soft prefix, will gradually replace with discrete
        current_visual = soft_prefix.detach().clone()

        for i in range(prefix_len):
            # Step 1: Compute FRESH gradient at current state
            current_visual_var = current_visual.detach().clone()
            current_visual_var.requires_grad = True

            loss, _ = self.attack.compute_loss(
                current_visual_var,
                victim_prompt,
                target_image_tokens,
                instruction_text,
                match_image_len,
            )
            loss.backward()
            fresh_grad = current_visual_var.grad

            # Step 2: HotFlip scoring - find tokens along negative gradient
            image_embeds = embed_weight[self.image_token_ids].float()
            token_scores = torch.matmul(
                image_embeds,
                -fresh_grad[0, i, :].float()
            )

            # Get top-k candidates
            top_k_limited = min(top_k, len(token_scores))
            top_indices_in_vocab = torch.topk(token_scores, top_k_limited).indices.tolist()
            top_indices = [self.image_token_ids[idx] for idx in top_indices_in_vocab]

            # Also include nearest neighbor by cosine similarity
            with torch.no_grad():
                sim_scores = F.cosine_similarity(
                    current_visual[0, i, :].unsqueeze(0),
                    image_embeds,
                    dim=1
                )
                nearest_idx = torch.argmax(sim_scores).item()
                nearest_token = self.image_token_ids[nearest_idx]

            if nearest_token not in top_indices:
                top_indices.append(nearest_token)

            # Step 3: Evaluate candidates
            best_loss = float('inf')
            best_token = top_indices[0]

            candidates_to_test = top_indices[:batch_size]

            for cand_id in candidates_to_test:
                visual_probe = current_visual.clone()
                cand_emb = self.embedding(
                    torch.tensor([[cand_id]], device=self.device)
                )
                visual_probe[0, i, :] = cand_emb.to(visual_probe.dtype)

                with torch.no_grad():
                    cand_loss, _ = self.attack.compute_loss(
                        visual_probe,
                        victim_prompt,
                        target_image_tokens,
                        instruction_text,
                        match_image_len,
                    )

                if cand_loss.item() < best_loss:
                    best_loss = cand_loss.item()
                    best_token = cand_id

            discrete_ids.append(best_token)

            # Step 4: Update current embedding for next iteration
            with torch.no_grad():
                best_emb = self.embedding(
                    torch.tensor([[best_token]], device=self.device)
                )
                current_visual[0, i, :] = best_emb.to(current_visual.dtype)

            # Progress logging
            if (i + 1) % 4 == 0 or i == 0:
                print(f"      Position {i+1}/{prefix_len}: "
                      f"Token {best_token} | Loss={best_loss:.4f}")

        # Calculate similarity to original soft prefix
        with torch.no_grad():
            discrete_emb = self.embedding(
                torch.tensor([discrete_ids], device=self.device)
            )
            similarity = F.cosine_similarity(
                soft_prefix.flatten(),
                discrete_emb.flatten(),
                dim=0
            ).item()

        print(f"\n    Hardening complete.")
        print(f"    Similarity to soft prefix: {similarity:.4f}")

        return discrete_ids
