#!/usr/bin/env python3
"""
Visual Adversarial Prefix Attack (视觉对抗前缀攻击)

核心突破：
- 不使用文本tokens，直接使用图像tokens
- Prompt结构: [BOI] [Adv_Image_Tokens] [EOI] "a photo of a cat"
- 搜索空间: Image Vocabulary (8195-16387)
- 优势: 同模态攻击，避开文本→图像的模态鸿沟

这是最有希望成功的方向，因为：
1. 直接在视觉流形中操作
2. 梯度指引精准
3. 避免语言瓶颈
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


class VisualPrefixAttack:
    """
    视觉对抗前缀攻击

    关键创新：
    - 使用图像tokens而非文本tokens
    - 避开跨模态鸿沟
    - 直接在视觉特征空间优化
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        visual_prefix_length: int = 16,  # 图像前缀长度（16-32个tokens）
    ):
        self.device = device
        self.visual_prefix_length = visual_prefix_length

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

        # 关键：获取图像vocabulary的范围
        self.image_token_ids = list(self.model.model.vocabulary_mapping.image_token_ids)
        print(f"Image token range: {min(self.image_token_ids)} - {max(self.image_token_ids)}")
        print(f"Total image tokens: {len(self.image_token_ids)}")

        print(f"Model loaded. Visual prefix length: {self.visual_prefix_length}")

    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本为token IDs"""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    def init_visual_soft_prefix(
        self,
        target_image_tokens: List[int],
        strategy: str = "target_semantic",
    ) -> nn.Parameter:
        """
        初始化视觉soft prefix

        策略：
        - target_semantic: 使用目标图像的前N个token的平均embedding
        - random_visual: 从图像vocabulary中随机采样embedding的平均
        - zero: 零初始化
        """
        if strategy == "target_semantic":
            # 使用目标图像的前visual_prefix_length个token
            sample_tokens = target_image_tokens[:self.visual_prefix_length]
            sample_ids = torch.tensor([sample_tokens], device=self.device, dtype=torch.long)
            with torch.no_grad():
                sample_emb = self.embedding(sample_ids)
                # 添加少量噪声以避免局部最优
                noise = torch.randn_like(sample_emb) * 0.02
                base = sample_emb + noise
        elif strategy == "random_visual":
            # 从图像vocabulary随机采样
            random_img_tokens = torch.randint(
                min(self.image_token_ids),
                max(self.image_token_ids) + 1,
                (1, self.visual_prefix_length),
                device=self.device
            )
            with torch.no_grad():
                base = self.embedding(random_img_tokens)
                noise = torch.randn_like(base) * 0.02
                base = base + noise
        else:  # zero
            base = torch.zeros(1, self.visual_prefix_length, self.hidden_size, device=self.device)

        param = nn.Parameter(base.float())
        param.requires_grad = True
        return param

    def compute_visual_attack_loss(
        self,
        visual_soft_prefix: torch.Tensor,
        victim_prompt: str,
        target_image_tokens: List[int],
        match_image_len: int = 128,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算视觉攻击损失

        Prompt结构: [BOI] [visual_soft_prefix] [EOI] [victim_prompt] [BOI] [target_image] [EOI]

        目标：让模型在看到victim prompt后，生成target image
        """
        # 构建prompt
        victim_ids = self.encode_text(victim_prompt)

        # BOI, EOI tokens
        boi = torch.tensor([[self.boi_id]], device=self.device, dtype=torch.long)
        eoi = torch.tensor([[self.eoi_id]], device=self.device, dtype=torch.long)

        # 目标图像tokens
        target_img_ids = torch.tensor([target_image_tokens[:match_image_len]], device=self.device, dtype=torch.long)

        # 获取embeddings
        victim_embeds = self.embedding(victim_ids)
        boi_embeds = self.embedding(boi)
        eoi_embeds = self.embedding(eoi)

        with torch.no_grad():
            target_img_embeds = self.embedding(target_img_ids)

        # 构建完整输入: [BOI] [visual_soft] [EOI] [victim] [BOI] [target_img] [EOI]
        input_embeds = torch.cat([
            boi_embeds,
            visual_soft_prefix.to(self.model.dtype),
            eoi_embeds,
            victim_embeds,
            boi_embeds,
            target_img_embeds,
            eoi_embeds
        ], dim=1)

        # 前向传播
        outputs = self.model.model(
            inputs_embeds=input_embeds,
            use_cache=False,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # 计算位置
        visual_len = visual_soft_prefix.shape[1]
        prefix_context_len = 1 + visual_len + 1 + victim_ids.shape[1] + 1  # BOI + visual + EOI + victim + BOI

        # 图像生成loss
        image_logits = logits[:, prefix_context_len-1:prefix_context_len-1+match_image_len, :]
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

    def optimize_visual_prefix(
        self,
        victim_prompt: str,
        target_image_tokens: List[int],
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        match_image_len: int = 128,
        init_strategy: str = "target_semantic",
        output_dir: str = "outputs/visual_prefix",
    ) -> Tuple[nn.Parameter, Dict]:
        """
        优化视觉对抗前缀
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print("Visual Adversarial Prefix Attack")
        print(f"{'='*70}")
        print(f"Victim prompt: '{victim_prompt}'")
        print(f"Visual prefix length: {self.visual_prefix_length}")
        print(f"Match image length: {match_image_len}")
        print(f"{'='*70}\n")

        # 初始化visual soft prefix
        visual_soft = self.init_visual_soft_prefix(
            target_image_tokens,
            strategy=init_strategy,
        )
        print(f"Visual soft prefix shape: {visual_soft.shape}")

        # 优化器
        optimizer = torch.optim.AdamW([visual_soft], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # 训练
        best_loss = float('inf')
        best_prefix = None
        history = []

        print("\nStarting optimization...")
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()

            loss, loss_dict = self.compute_visual_attack_loss(
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
                best_prefix = visual_soft.detach().clone()

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d} | Loss: {loss_dict['total']:.4f}")

            if epoch % 50 == 0:
                torch.cuda.empty_cache()

        # 保存
        final_prefix = best_prefix if best_prefix is not None else visual_soft.detach()

        checkpoint = {
            'visual_soft_prefix': final_prefix.cpu(),
            'best_loss': best_loss,
            'history': history,
            'config': {
                'victim_prompt': victim_prompt,
                'visual_prefix_length': self.visual_prefix_length,
                'match_image_len': match_image_len,
                'init_strategy': init_strategy,
                'num_epochs': num_epochs,
            }
        }

        save_path = os.path.join(output_dir, "visual_soft_prefix.pt")
        torch.save(checkpoint, save_path)
        print(f"\n✓ Saved to: {save_path}")

        return final_prefix, {'best_loss': best_loss, 'history': history}

    def gradient_guided_visual_hardening(
        self,
        visual_soft: torch.Tensor,
        victim_prompt: str,
        target_image_tokens: List[int],
        match_image_len: int = 128,
        top_k: int = 64,
        batch_size: int = 32,
    ) -> List[int]:
        """
        梯度引导的视觉硬化

        关键：搜索空间限制在图像vocabulary (8195-16387)
        """
        discrete_ids = []
        embed_weight = self.embedding.weight

        print(f"\n    Running Visual Gradient-Guided Hardening...")
        print(f"    Search space: Image tokens {min(self.image_token_ids)}-{max(self.image_token_ids)}")

        current_visual = visual_soft.detach().clone()

        for i in range(visual_soft.shape[1]):
            # 重新计算梯度
            current_visual_var = current_visual.detach().clone()
            current_visual_var.requires_grad = True

            loss, _ = self.compute_visual_attack_loss(
                current_visual_var, victim_prompt, target_image_tokens, match_image_len
            )
            loss.backward()
            fresh_grad = current_visual_var.grad

            # HotFlip: 只在图像vocabulary中搜索
            image_embeds = embed_weight[self.image_token_ids].float()
            token_scores = torch.matmul(image_embeds, -fresh_grad[0, i, :].float())

            # Top-K
            top_k_limited = min(top_k, len(token_scores))
            top_indices_in_img_vocab = torch.topk(token_scores, top_k_limited).indices.tolist()

            # 转换为实际token IDs
            top_indices = [self.image_token_ids[idx] for idx in top_indices_in_img_vocab]

            # 几何最近邻保底
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

            # 真实评估
            best_token_loss = float('inf')
            best_token_id = top_indices[0]

            candidates_to_test = top_indices[:batch_size]

            for cand_id in candidates_to_test:
                visual_probe = current_visual.clone()
                cand_emb = self.embedding(torch.tensor([[cand_id]], device=self.device))
                visual_probe[0, i, :] = cand_emb.to(visual_probe.dtype)

                with torch.no_grad():
                    l, _ = self.compute_visual_attack_loss(
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
    def verify_visual_attack(
        self,
        visual_discrete_ids: List[int],
        victim_prompt: str,
        max_new_tokens: int = 1200,
    ) -> Tuple[str, List[int], List[int]]:
        """
        验证视觉对抗前缀攻击
        """
        print("\n" + "="*70)
        print("Verifying Visual Prefix Attack")
        print("="*70)

        # 构建输入: [BOI] [visual_prefix] [EOI] [victim]
        boi = torch.tensor([[self.boi_id]], device=self.device, dtype=torch.long)
        eoi = torch.tensor([[self.eoi_id]], device=self.device, dtype=torch.long)
        visual_ids = torch.tensor([visual_discrete_ids], device=self.device, dtype=torch.long)
        victim_ids = self.encode_text(victim_prompt)

        input_ids = torch.cat([boi, visual_ids, eoi, victim_ids], dim=1)
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

        # 解析
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
    parser = argparse.ArgumentParser(description="Visual Adversarial Prefix Attack")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--target_tokens_file", type=str, required=True)
    parser.add_argument("--victim_prompt", type=str, default="a photo of a cat")
    parser.add_argument("--visual_prefix_length", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--match_image_len", type=int, default=128)
    parser.add_argument("--init_strategy", type=str, default="target_semantic",
                        choices=["target_semantic", "random_visual", "zero"])
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(REPO_ROOT, "outputs", "visual_prefix"))
    parser.add_argument("--harden", action="store_true")
    parser.add_argument("--verify", action="store_true")

    args = parser.parse_args()

    # 加载目标tokens
    with open(args.target_tokens_file, 'r') as f:
        target_data = json.load(f)
    target_tokens = target_data['image_tokens_list'][0]
    print(f"Loaded {len(target_tokens)} target image tokens")

    # 初始化攻击
    attack = VisualPrefixAttack(
        model_path=args.model_path,
        visual_prefix_length=args.visual_prefix_length,
    )

    # 优化visual soft prefix
    visual_soft, results = attack.optimize_visual_prefix(
        victim_prompt=args.victim_prompt,
        target_image_tokens=target_tokens,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        match_image_len=args.match_image_len,
        init_strategy=args.init_strategy,
        output_dir=args.output_dir,
    )

    # 硬化
    if args.harden:
        print(f"\n{'='*70}")
        print("Hardening Visual Prefix")
        print(f"{'='*70}")

        visual_discrete = attack.gradient_guided_visual_hardening(
            visual_soft.to(attack.device),
            args.victim_prompt,
            target_tokens,
            args.match_image_len,
        )

        harden_results = {
            'visual_discrete_ids': visual_discrete,
            'visual_prefix_length': len(visual_discrete),
        }

        harden_path = os.path.join(args.output_dir, "visual_discrete_prefix.json")
        with open(harden_path, 'w') as f:
            json.dump(harden_results, f, indent=2)
        print(f"\nSaved hardened prefix to: {harden_path}")

        # 验证
        if args.verify:
            text_out, full_tokens, image_tokens = attack.verify_visual_attack(
                visual_discrete,
                args.victim_prompt,
            )

            verify_results = {
                'visual_prefix_ids': visual_discrete,
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
