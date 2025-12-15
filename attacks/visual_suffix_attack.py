#!/usr/bin/env python3
"""
Visual Suffix Attack (视觉后缀攻击)

核心思路：
- Prompt结构: "a photo of a cat" [BOI] [Adv_Visual_Suffix] [EOI]
- 利用近因效应：模型最后看到的是adversarial visual suffix
- 可能绕过victim prompt的主导地位
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple, Dict

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


class VisualSuffixAttack:
    """视觉后缀攻击 - 利用近因效应"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        visual_suffix_length: int = 16,
    ):
        self.device = device
        self.visual_suffix_length = visual_suffix_length

        print("Loading model...")
        self.config = ChameleonConfig.from_pretrained(model_path)
        self.config._attn_implementation = "eager"

        self.processor = ChameleonProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = ChameleonForConditionalGenerationWithCFG.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=dtype,
        ).to(device)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.train()
        self.model.gradient_checkpointing_enable()
        self.model.setup_cfg(cfg_config="no")

        self.embedding = self.model.model.embed_tokens
        self.lm_head = self.model.lm_head
        self.boi_id = self.config.boi_token_id
        self.eoi_id = self.config.eoi_token_id
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

        self.image_token_ids = list(self.model.model.vocabulary_mapping.image_token_ids)
        print(f"Image token range: {min(self.image_token_ids)} - {max(self.image_token_ids)}")
        print(f"Model loaded. Visual suffix length: {self.visual_suffix_length}")

    def encode_text(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    def init_visual_soft_suffix(
        self,
        target_image_tokens: List[int],
        strategy: str = "target_semantic",
    ) -> nn.Parameter:
        if strategy == "target_semantic":
            sample_tokens = target_image_tokens[:self.visual_suffix_length]
            sample_ids = torch.tensor([sample_tokens], device=self.device, dtype=torch.long)
            with torch.no_grad():
                sample_emb = self.embedding(sample_ids)
                noise = torch.randn_like(sample_emb) * 0.02
                base = sample_emb + noise
        elif strategy == "random_visual":
            random_img_tokens = torch.randint(
                min(self.image_token_ids),
                max(self.image_token_ids) + 1,
                (1, self.visual_suffix_length),
                device=self.device
            )
            with torch.no_grad():
                base = self.embedding(random_img_tokens)
                noise = torch.randn_like(base) * 0.02
                base = base + noise
        else:
            base = torch.zeros(1, self.visual_suffix_length, self.hidden_size, device=self.device)

        param = nn.Parameter(base.float())
        param.requires_grad = True
        return param

    def compute_visual_suffix_loss(
        self,
        visual_soft_suffix: torch.Tensor,
        victim_prompt: str,
        target_image_tokens: List[int],
        match_image_len: int = 128,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算视觉后缀损失

        Prompt结构: [victim_prompt] [BOI] [visual_soft_suffix] [EOI] [target_output]

        关键：suffix紧接在生成开始前，利用近因效应
        """
        victim_ids = self.encode_text(victim_prompt)

        boi = torch.tensor([[self.boi_id]], device=self.device, dtype=torch.long)
        eoi = torch.tensor([[self.eoi_id]], device=self.device, dtype=torch.long)

        target_img_ids = torch.tensor([target_image_tokens[:match_image_len]], device=self.device, dtype=torch.long)

        victim_embeds = self.embedding(victim_ids)
        boi_embeds = self.embedding(boi)
        eoi_embeds = self.embedding(eoi)

        with torch.no_grad():
            target_img_embeds = self.embedding(target_img_ids)

        # 构建: [victim] [BOI] [visual_soft_suffix] [EOI] [BOI] [target_img] [EOI]
        # 注意：suffix在victim之后、目标生成之前
        input_embeds = torch.cat([
            victim_embeds,
            boi_embeds,
            visual_soft_suffix.to(self.model.dtype),
            eoi_embeds,
            boi_embeds,
            target_img_embeds,
            eoi_embeds
        ], dim=1)

        outputs = self.model.model(
            inputs_embeds=input_embeds,
            use_cache=False,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # 计算位置
        victim_len = victim_ids.shape[1]
        suffix_len = visual_soft_suffix.shape[1]
        context_len = victim_len + 1 + suffix_len + 1 + 1  # victim + BOI + suffix + EOI + BOI

        # 图像生成loss
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

    def optimize_visual_suffix(
        self,
        victim_prompt: str,
        target_image_tokens: List[int],
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        match_image_len: int = 128,
        init_strategy: str = "target_semantic",
        output_dir: str = "outputs/visual_suffix",
    ) -> Tuple[nn.Parameter, Dict]:
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print("Visual Suffix Attack (Recency Bias Exploitation)")
        print(f"{'='*70}")
        print(f"Victim prompt: '{victim_prompt}'")
        print(f"Visual suffix length: {self.visual_suffix_length}")
        print(f"Strategy: Suffix comes AFTER victim prompt")
        print(f"{'='*70}\n")

        visual_soft = self.init_visual_soft_suffix(
            target_image_tokens,
            strategy=init_strategy,
        )
        print(f"Visual soft suffix shape: {visual_soft.shape}")

        optimizer = torch.optim.AdamW([visual_soft], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_loss = float('inf')
        best_suffix = None
        history = []

        print("\nStarting optimization...")
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()

            loss, loss_dict = self.compute_visual_suffix_loss(
                visual_soft,
                victim_prompt,
                target_image_tokens,
                match_image_len,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([visual_soft], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            history.append(loss_dict)

            if loss_dict['total'] < best_loss:
                best_loss = loss_dict['total']
                best_suffix = visual_soft.detach().clone()

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d} | Loss: {loss_dict['total']:.4f}")

            if epoch % 50 == 0:
                torch.cuda.empty_cache()

        final_suffix = best_suffix if best_suffix is not None else visual_soft.detach()

        checkpoint = {
            'visual_soft_suffix': final_suffix.cpu(),
            'best_loss': best_loss,
            'history': history,
            'config': {
                'victim_prompt': victim_prompt,
                'visual_suffix_length': self.visual_suffix_length,
                'match_image_len': match_image_len,
                'init_strategy': init_strategy,
                'num_epochs': num_epochs,
            }
        }

        save_path = os.path.join(output_dir, "visual_soft_suffix.pt")
        torch.save(checkpoint, save_path)
        print(f"\n✓ Saved to: {save_path}")

        return final_suffix, {'best_loss': best_loss, 'history': history}

    def gradient_guided_visual_hardening(
        self,
        visual_soft: torch.Tensor,
        victim_prompt: str,
        target_image_tokens: List[int],
        match_image_len: int = 128,
        top_k: int = 64,
        batch_size: int = 32,
    ) -> List[int]:
        discrete_ids = []
        embed_weight = self.embedding.weight

        print(f"\n    Running Visual Suffix Gradient-Guided Hardening...")

        current_visual = visual_soft.detach().clone()

        for i in range(visual_soft.shape[1]):
            current_visual_var = current_visual.detach().clone()
            current_visual_var.requires_grad = True

            loss, _ = self.compute_visual_suffix_loss(
                current_visual_var, victim_prompt, target_image_tokens, match_image_len
            )
            loss.backward()
            fresh_grad = current_visual_var.grad

            image_embeds = embed_weight[self.image_token_ids].float()
            token_scores = torch.matmul(image_embeds, -fresh_grad[0, i, :].float())

            top_k_limited = min(top_k, len(token_scores))
            top_indices_in_img_vocab = torch.topk(token_scores, top_k_limited).indices.tolist()
            top_indices = [self.image_token_ids[idx] for idx in top_indices_in_img_vocab]

            with torch.no_grad():
                sim_scores = F.cosine_similarity(
                    current_visual[0, i, :].unsqueeze(0),
                    image_embeds,
                    dim=1
                )
                nearest_in_img_vocab = torch.argmax(sim_scores).item()
                nearest_token = self.image_token_ids[nearest_in_img_vocab]

            if nearest_token not in top_indices:
                top_indices.append(nearest_token)

            best_token_loss = float('inf')
            best_token_id = top_indices[0]

            candidates_to_test = top_indices[:batch_size]

            for cand_id in candidates_to_test:
                visual_probe = current_visual.clone()
                cand_emb = self.embedding(torch.tensor([[cand_id]], device=self.device))
                visual_probe[0, i, :] = cand_emb.to(visual_probe.dtype)

                with torch.no_grad():
                    l, _ = self.compute_visual_suffix_loss(
                        visual_probe, victim_prompt, target_image_tokens, match_image_len
                    )

                if l.item() < best_token_loss:
                    best_token_loss = l.item()
                    best_token_id = cand_id

            discrete_ids.append(best_token_id)

            with torch.no_grad():
                best_emb = self.embedding(torch.tensor([[best_token_id]], device=self.device))
                current_visual[0, i, :] = best_emb.to(current_visual.dtype)

            if (i + 1) % 4 == 0 or i == 0:
                print(f"      Position {i+1}/{visual_soft.shape[1]}: ImgToken_{best_token_id} | Loss={best_token_loss:.4f}")

        return discrete_ids

    @torch.no_grad()
    def verify_visual_suffix_attack(
        self,
        visual_discrete_ids: List[int],
        victim_prompt: str,
        max_new_tokens: int = 1200,
    ) -> Tuple[str, List[int], List[int]]:
        print("\n" + "="*70)
        print("Verifying Visual Suffix Attack")
        print("="*70)

        # 构建: [victim] [BOI] [visual_suffix] [EOI]
        victim_ids = self.encode_text(victim_prompt)
        boi = torch.tensor([[self.boi_id]], device=self.device, dtype=torch.long)
        eoi = torch.tensor([[self.eoi_id]], device=self.device, dtype=torch.long)
        visual_ids = torch.tensor([visual_discrete_ids], device=self.device, dtype=torch.long)

        input_ids = torch.cat([victim_ids, boi, visual_ids, eoi], dim=1)
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
        print("="*70)

        return text_output, generated_ids, image_tokens


def main():
    parser = argparse.ArgumentParser(description="Visual Suffix Attack")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--target_tokens_file", type=str, required=True)
    parser.add_argument("--victim_prompt", type=str, default="a photo of a cat")
    parser.add_argument("--visual_suffix_length", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--match_image_len", type=int, default=128)
    parser.add_argument("--init_strategy", type=str, default="target_semantic")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(REPO_ROOT, "outputs", "visual_suffix"))
    parser.add_argument("--harden", action="store_true")
    parser.add_argument("--verify", action="store_true")

    args = parser.parse_args()

    with open(args.target_tokens_file, 'r') as f:
        target_data = json.load(f)
    target_tokens = target_data['image_tokens_list'][0]
    print(f"Loaded {len(target_tokens)} target image tokens")

    attack = VisualSuffixAttack(
        model_path=args.model_path,
        visual_suffix_length=args.visual_suffix_length,
    )

    visual_soft, results = attack.optimize_visual_suffix(
        victim_prompt=args.victim_prompt,
        target_image_tokens=target_tokens,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        match_image_len=args.match_image_len,
        init_strategy=args.init_strategy,
        output_dir=args.output_dir,
    )

    if args.harden:
        print(f"\n{'='*70}")
        print("Hardening Visual Suffix")
        print(f"{'='*70}")

        visual_discrete = attack.gradient_guided_visual_hardening(
            visual_soft.to(attack.device),
            args.victim_prompt,
            target_tokens,
            args.match_image_len,
        )

        harden_results = {
            'visual_discrete_ids': visual_discrete,
            'visual_suffix_length': len(visual_discrete),
        }

        harden_path = os.path.join(args.output_dir, "visual_discrete_suffix.json")
        with open(harden_path, 'w') as f:
            json.dump(harden_results, f, indent=2)
        print(f"\nSaved hardened suffix to: {harden_path}")

        if args.verify:
            text_out, full_tokens, image_tokens = attack.verify_visual_suffix_attack(
                visual_discrete,
                args.victim_prompt,
            )

            verify_results = {
                'visual_suffix_ids': visual_discrete,
                'generated_text': text_out,
                'full_tokens': full_tokens,
                'image_tokens': image_tokens,
                'num_image_tokens': len(image_tokens),
            }

            verify_path = os.path.join(args.output_dir, "verification_results.json")
            with open(verify_path, 'w') as f:
                json.dump(verify_results, f, indent=2)
            print(f"Saved verification to: {verify_path}")


if __name__ == "__main__":
    main()
