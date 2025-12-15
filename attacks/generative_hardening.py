#!/usr/bin/env python3
"""
Generative Hardening + GCG Refinement (生成式硬化 + GCG精化)

核心改进（基于LARGO思想）：
1. 不使用nearest neighbor硬化
2. 让模型"自反式解码"soft prompt的意图
3. 对生成的候选进行ranking
4. 用GCG对最佳候选进行精化（使用语言桥接loss）

流程：
Step 1: Self-Reflective Decoding
    - 输入: [soft_prefix] + "What will you generate?"
    - 模型输出: "I will generate a dog..."
    - 这是soft prompt编码的"意图"

Step 2: Candidate Ranking
    - 生成多个候选
    - 用语言桥接loss评分
    - 选择最佳候选作为初始离散prefix

Step 3: GCG Refinement
    - 对离散prefix逐位置优化
    - 目标: 最小化语言桥接loss
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def _prefer_local_transformers(repo_root: str) -> None:
    local_tf_src = os.path.join(repo_root, "transformers", "src")
    if os.path.isdir(local_tf_src) and local_tf_src not in sys.path:
        sys.path.insert(0, local_tf_src)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_prefer_local_transformers(REPO_ROOT)

from transformers import (
    ChameleonConfig,
    ChameleonProcessor,
    ChameleonForConditionalGenerationWithCFG,
)


class GenerativeHardening:
    """
    生成式硬化: 让模型自己将soft prompt转换为离散文本
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        self.device = device

        print("Loading model...")
        self.config = ChameleonConfig.from_pretrained(model_path)
        self.config._attn_implementation = "eager"

        self.processor = ChameleonProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        self.model = ChameleonForConditionalGenerationWithCFG.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
        ).to(device)

        self.model.eval()
        self.model.setup_cfg(cfg_config="no")

        self.embedding = self.model.model.embed_tokens
        self.lm_head = self.model.lm_head
        self.boi_id = self.config.boi_token_id
        self.eoi_id = self.config.eoi_token_id
        self.vocab_size = self.config.vocab_size

        print(f"Model loaded. Vocab size: {self.vocab_size}")

    def encode_text(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    @torch.no_grad()
    def self_reflective_decode(
        self,
        soft_prefix: torch.Tensor,
        reflection_prompt: str = "Based on my instructions, I will now",
        max_new_tokens: int = 30,
        temperature: float = 0.7,
        num_candidates: int = 5,
    ) -> List[Tuple[str, List[int]]]:
        """
        自反式解码: 让模型根据soft prefix生成它"理解"的意图

        Returns:
            candidates: List of (decoded_text, token_ids)
        """
        print(f"\nSelf-Reflective Decoding with prompt: '{reflection_prompt}'")

        reflection_ids = self.encode_text(reflection_prompt)
        reflection_embeds = self.embedding(reflection_ids)

        # [soft_prefix] + [reflection_prompt]
        input_embeds = torch.cat([
            soft_prefix.to(self.model.dtype).to(self.device),
            reflection_embeds
        ], dim=1)

        candidates = []

        for i in range(num_candidates):
            current_embeds = input_embeds.clone()
            generated_ids = []

            for step in range(max_new_tokens):
                outputs = self.model.model(
                    inputs_embeds=current_embeds,
                    use_cache=False,
                )

                logits = self.lm_head(outputs.last_hidden_state[:, -1:, :])

                # Temperature sampling
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs.squeeze(0).squeeze(0), 1).item()
                else:
                    next_id = torch.argmax(logits, dim=-1).item()

                generated_ids.append(next_id)

                # Stop at special tokens
                if next_id in [self.boi_id, self.eoi_id, self.tokenizer.eos_token_id]:
                    break

                next_emb = self.embedding(torch.tensor([[next_id]], device=self.device))
                current_embeds = torch.cat([current_embeds, next_emb.to(self.model.dtype)], dim=1)

            decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            candidates.append((decoded_text, generated_ids))

            print(f"  Candidate {i+1}: '{decoded_text}'")

        return candidates

    def compute_bridge_loss_for_prefix(
        self,
        prefix_ids: List[int],
        victim_prompt: str,
        bridge_text: str,
        target_image_tokens: List[int],
        match_image_len: int = 128,
    ) -> float:
        """
        计算给定离散prefix的语言桥接loss
        """
        # Build sequences
        prefix_tensor = torch.tensor([prefix_ids], device=self.device, dtype=torch.long)
        victim_ids = self.encode_text(victim_prompt)
        bridge_ids = self.encode_text(bridge_text)

        boi = torch.tensor([[self.boi_id]], device=self.device, dtype=torch.long)
        eoi = torch.tensor([[self.eoi_id]], device=self.device, dtype=torch.long)
        image_ids = torch.tensor([target_image_tokens[:match_image_len]], device=self.device, dtype=torch.long)

        # Target: [bridge] + [BOI] + [image] + [EOI]
        target_ids = torch.cat([bridge_ids, boi, image_ids, eoi], dim=1)

        # Input embeddings: [prefix] + [victim] + [target]
        prefix_embeds = self.embedding(prefix_tensor)
        victim_embeds = self.embedding(victim_ids)
        target_embeds = self.embedding(target_ids)

        input_embeds = torch.cat([
            prefix_embeds, victim_embeds, target_embeds
        ], dim=1).to(self.model.dtype)

        # Forward
        with torch.no_grad():
            outputs = self.model.model(
                inputs_embeds=input_embeds,
                use_cache=False,
            )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # Compute loss
        prefix_len = len(prefix_ids)
        victim_len = victim_ids.shape[1]
        context_len = prefix_len + victim_len
        bridge_len = bridge_ids.shape[1]

        # Bridge loss
        bridge_logits = logits[:, context_len-1:context_len+bridge_len-1, :]
        bridge_targets = target_ids[:, :bridge_len]

        min_len = min(bridge_logits.shape[1], bridge_targets.shape[1])
        loss = F.cross_entropy(
            bridge_logits[:, :min_len, :].reshape(-1, self.vocab_size),
            bridge_targets[:, :min_len].reshape(-1),
            reduction='mean'
        )

        return loss.item()

    def rank_candidates(
        self,
        candidates: List[Tuple[str, List[int]]],
        victim_prompt: str,
        bridge_text: str,
        target_image_tokens: List[int],
    ) -> List[Tuple[str, List[int], float]]:
        """
        对候选进行ranking
        """
        print("\nRanking candidates...")

        ranked = []
        for text, ids in candidates:
            loss = self.compute_bridge_loss_for_prefix(
                ids, victim_prompt, bridge_text, target_image_tokens
            )
            ranked.append((text, ids, loss))
            print(f"  '{text}' -> loss: {loss:.4f}")

        # Sort by loss (lower is better)
        ranked.sort(key=lambda x: x[2])
        return ranked

    def gcg_refine(
        self,
        initial_prefix_ids: List[int],
        victim_prompt: str,
        bridge_text: str,
        target_image_tokens: List[int],
        num_steps: int = 20,
        k_candidates: int = 16,
        match_image_len: int = 128,
    ) -> Tuple[List[int], float, List[float]]:
        """
        GCG精化: 逐位置优化离散prefix
        """
        print(f"\nGCG Refinement: {num_steps} steps, {k_candidates} candidates/position")

        current_ids = list(initial_prefix_ids)
        prefix_len = len(current_ids)

        initial_loss = self.compute_bridge_loss_for_prefix(
            current_ids, victim_prompt, bridge_text, target_image_tokens, match_image_len
        )
        print(f"Initial loss: {initial_loss:.4f}")

        best_loss = initial_loss
        best_ids = list(current_ids)
        loss_history = [initial_loss]

        # Get valid token range (exclude special tokens)
        image_token_ids = set(self.model.model.vocabulary_mapping.image_token_ids)
        special_ids = {self.boi_id, self.eoi_id, 0, 1, 2}
        excluded_ids = image_token_ids | special_ids

        valid_token_ids = [i for i in range(self.vocab_size) if i not in excluded_ids]

        for step in range(num_steps):
            improved = False

            for pos in range(prefix_len):
                # Sample random candidates
                candidates_ids = torch.randint(0, len(valid_token_ids), (k_candidates,))
                candidates = [valid_token_ids[i] for i in candidates_ids]

                pos_best_loss = best_loss
                pos_best_token = current_ids[pos]

                for cand_id in candidates:
                    # Create variant
                    variant_ids = list(current_ids)
                    variant_ids[pos] = cand_id

                    # Evaluate
                    loss = self.compute_bridge_loss_for_prefix(
                        variant_ids, victim_prompt, bridge_text,
                        target_image_tokens, match_image_len
                    )

                    if loss < pos_best_loss:
                        pos_best_loss = loss
                        pos_best_token = cand_id

                # Update if improved
                if pos_best_loss < best_loss:
                    old_token = current_ids[pos]
                    current_ids[pos] = pos_best_token
                    best_loss = pos_best_loss
                    best_ids = list(current_ids)
                    improved = True

                    old_str = self.tokenizer.decode([old_token])
                    new_str = self.tokenizer.decode([pos_best_token])
                    print(f"  Step {step+1}, Pos {pos}: '{old_str}' -> '{new_str}' | Loss: {best_loss:.4f}")

            loss_history.append(best_loss)

            if not improved:
                print(f"  Step {step+1}: No improvement")

        print(f"\nGCG Complete: {initial_loss:.4f} -> {best_loss:.4f} ({(1-best_loss/initial_loss)*100:.1f}% reduction)")

        return best_ids, best_loss, loss_history

    @torch.no_grad()
    def verify_discrete_prefix(
        self,
        prefix_ids: List[int],
        victim_prompt: str,
        max_new_tokens: int = 1200,
    ) -> Tuple[str, List[int], List[int]]:
        """
        验证离散prefix的攻击效果
        """
        print("\nVerifying discrete prefix...")

        prefix_tensor = torch.tensor([prefix_ids], device=self.device, dtype=torch.long)
        victim_ids = self.encode_text(victim_prompt)

        input_ids = torch.cat([prefix_tensor, victim_ids], dim=1)
        current_embeds = self.embedding(input_ids).to(self.model.dtype)

        generated_ids = []

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

        text_output = self.tokenizer.decode(text_tokens, skip_special_tokens=True)

        print(f"Generated text: '{text_output}'")
        print(f"Generated {len(image_tokens)} image tokens")

        return text_output, generated_ids, image_tokens


def main():
    parser = argparse.ArgumentParser(description="Generative Hardening + GCG")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--soft_prompt_file", type=str, required=True,
                        help="Path to language bridge soft prompt checkpoint")
    parser.add_argument("--target_tokens_file", type=str, required=True)
    parser.add_argument("--victim_prompt", type=str, default="a photo of a cat")
    parser.add_argument("--bridge_text", type=str,
                        default="Sure, I'll generate a cute dog for you.")
    parser.add_argument("--reflection_prompt", type=str,
                        default="Based on my instructions, I will now")
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--gcg_steps", type=int, default=20)
    parser.add_argument("--gcg_k", type=int, default=32)
    parser.add_argument("--match_image_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(REPO_ROOT, "outputs", "generative_hardening"))
    parser.add_argument("--verify", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load soft prompt
    print(f"Loading soft prompt from: {args.soft_prompt_file}")
    checkpoint = torch.load(args.soft_prompt_file, map_location="cpu")
    soft_prefix = checkpoint['soft_prefix']
    print(f"Soft prefix shape: {soft_prefix.shape}")

    # Load target tokens
    with open(args.target_tokens_file, 'r') as f:
        target_data = json.load(f)
    target_tokens = target_data['image_tokens_list'][0]
    print(f"Loaded {len(target_tokens)} target image tokens")

    # Initialize
    hardener = GenerativeHardening(model_path=args.model_path)

    print(f"\n{'='*70}")
    print("Step 1: Self-Reflective Decoding")
    print(f"{'='*70}")

    # Self-reflective decode
    candidates = hardener.self_reflective_decode(
        soft_prefix=soft_prefix,
        reflection_prompt=args.reflection_prompt,
        num_candidates=args.num_candidates,
    )

    print(f"\n{'='*70}")
    print("Step 2: Candidate Ranking")
    print(f"{'='*70}")

    # Rank candidates
    ranked = hardener.rank_candidates(
        candidates,
        args.victim_prompt,
        args.bridge_text,
        target_tokens,
    )

    # Select best candidate
    best_text, best_ids, best_loss = ranked[0]
    print(f"\nBest candidate: '{best_text}' (loss: {best_loss:.4f})")

    print(f"\n{'='*70}")
    print("Step 3: GCG Refinement")
    print(f"{'='*70}")

    # GCG refine
    refined_ids, final_loss, loss_history = hardener.gcg_refine(
        initial_prefix_ids=best_ids,
        victim_prompt=args.victim_prompt,
        bridge_text=args.bridge_text,
        target_image_tokens=target_tokens,
        num_steps=args.gcg_steps,
        k_candidates=args.gcg_k,
        match_image_len=args.match_image_len,
    )

    refined_text = hardener.tokenizer.decode(refined_ids)
    print(f"\nFinal refined prefix: '{refined_text}'")

    # Save results
    results = {
        'refined_prefix_ids': refined_ids,
        'refined_prefix_text': refined_text,
        'initial_candidate': best_text,
        'initial_candidate_ids': best_ids,
        'initial_loss': best_loss,
        'final_loss': final_loss,
        'loss_history': loss_history,
        'all_candidates': [(t, ids, l) for t, ids, l in ranked],
        'config': {
            'victim_prompt': args.victim_prompt,
            'bridge_text': args.bridge_text,
            'gcg_steps': args.gcg_steps,
            'gcg_k': args.gcg_k,
        }
    }

    results_path = os.path.join(args.output_dir, "generative_hardening_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Verify
    if args.verify:
        print(f"\n{'='*70}")
        print("Verification")
        print(f"{'='*70}")

        text_out, full_tokens, image_tokens = hardener.verify_discrete_prefix(
            refined_ids,
            args.victim_prompt,
        )

        verify_results = {
            'prefix_text': refined_text,
            'generated_text': text_out,
            'full_tokens': full_tokens,
            'image_tokens': image_tokens,
            'num_image_tokens': len(image_tokens),
            'has_boi': hardener.boi_id in full_tokens,
            'has_eoi': hardener.eoi_id in full_tokens,
        }

        verify_path = os.path.join(args.output_dir, "verification_results.json")
        with open(verify_path, 'w') as f:
            json.dump(verify_results, f, indent=2)
        print(f"Saved verification to: {verify_path}")


if __name__ == "__main__":
    main()
