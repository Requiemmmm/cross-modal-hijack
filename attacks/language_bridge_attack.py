#!/usr/bin/env python3
"""
Language Bridge Attack (语言桥接攻击)

核心改进（基于LARGO思想的多模态适配）：
1. 语言桥接：让模型先说"我将生成一只狗"再生成图像
2. 语言意图损失：L_lang确保soft prompt学到可解释的语义
3. 多阶段优化：先语言意图，再图像生成
4. 对比硬化：使离散prompt更像目标，更不像原始

目标序列结构：
[soft_prefix] + [victim_prompt] → [bridge_text] + [BOI] + [target_image] + [EOI]

例如：
[prefix] + "a photo of a cat" → "Sure, I'll generate a dog for you." + [BOI] + [dog_tokens] + [EOI]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple, Optional, Dict

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


class LanguageBridgeAttack:
    """
    语言桥接攻击：通过语言意图引导实现稳定的跨模态劫持

    核心思想：让模型在文本层先"说服自己"，再生成目标图像
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        prefix_length: int = 20,
    ):
        self.device = device
        self.prefix_length = prefix_length

        print("Loading model and processor...")
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

        # Freeze model
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.train()
        self.model.gradient_checkpointing_enable()
        self.model.setup_cfg(cfg_config="no")

        # Get components
        self.embedding = self.model.model.embed_tokens
        self.lm_head = self.model.lm_head
        self.boi_id = self.config.boi_token_id
        self.eoi_id = self.config.eoi_token_id
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

        print(f"Model loaded. Hidden size: {self.hidden_size}")
        print(f"BOI: {self.boi_id}, EOI: {self.eoi_id}")

    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本为token IDs"""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    def init_soft_prefix(
        self,
        strategy: str = "bridge_semantic",
        bridge_text: str = None,
        target_text: str = None,
    ) -> nn.Parameter:
        """
        初始化soft prefix

        策略:
        - random: 随机初始化
        - bridge_semantic: 从桥接文本的语义初始化（推荐）
        - target_semantic: 从目标文本初始化
        """
        if strategy == "bridge_semantic" and bridge_text:
            # 使用桥接文本的平均embedding
            bridge_ids = self.encode_text(bridge_text)
            with torch.no_grad():
                bridge_emb = self.embedding(bridge_ids)
                mean_emb = bridge_emb.mean(dim=1, keepdim=True)
                base = mean_emb.repeat(1, self.prefix_length, 1)
        elif strategy == "target_semantic" and target_text:
            target_ids = self.encode_text(target_text)
            with torch.no_grad():
                target_emb = self.embedding(target_ids)
                mean_emb = target_emb.mean(dim=1, keepdim=True)
                base = mean_emb.repeat(1, self.prefix_length, 1)
        else:
            base = torch.randn(1, self.prefix_length, self.hidden_size, device=self.device) * 0.02

        noise = torch.randn_like(base) * 0.01
        param = nn.Parameter((base + noise).float())
        param.requires_grad = True
        return param

    def build_target_sequence(
        self,
        bridge_text: str,
        target_image_tokens: List[int],
    ) -> Tuple[torch.Tensor, int, int]:
        """
        构建目标序列: [bridge_text] + [BOI] + [image_tokens] + [EOI]

        Returns:
            target_ids: 完整目标token序列
            bridge_end_pos: 桥接文本结束位置
            image_start_pos: 图像开始位置（BOI之后）
        """
        bridge_ids = self.encode_text(bridge_text)
        bridge_len = bridge_ids.shape[1]

        boi = torch.tensor([[self.boi_id]], device=self.device, dtype=torch.long)
        eoi = torch.tensor([[self.eoi_id]], device=self.device, dtype=torch.long)
        image_ids = torch.tensor([target_image_tokens], device=self.device, dtype=torch.long)

        # [bridge] + [BOI] + [image] + [EOI]
        target_ids = torch.cat([bridge_ids, boi, image_ids, eoi], dim=1)

        return target_ids, bridge_len, bridge_len + 1

    def compute_language_bridge_loss(
        self,
        soft_prefix: torch.Tensor,
        victim_ids: torch.Tensor,
        target_ids: torch.Tensor,
        bridge_end_pos: int,
        image_start_pos: int,
        lambda_bridge: float = 2.0,
        lambda_image: float = 1.0,
        match_image_len: int = 256,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算语言桥接损失

        Loss = λ_bridge * L_bridge + λ_image * L_image

        L_bridge: 桥接文本的交叉熵损失（让模型说出目标意图）
        L_image: 图像token的交叉熵损失
        """
        # 获取embeddings
        victim_embeds = self.embedding(victim_ids)

        with torch.no_grad():
            target_embeds = self.embedding(target_ids)

        # 构建输入: [prefix] + [victim] + [target]（teacher forcing）
        input_embeds = torch.cat([
            soft_prefix.to(self.model.dtype),
            victim_embeds,
            target_embeds
        ], dim=1)

        # 前向传播
        outputs = self.model.model(
            inputs_embeds=input_embeds,
            use_cache=False,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # 计算位置
        prefix_len = soft_prefix.shape[1]
        victim_len = victim_ids.shape[1]
        context_len = prefix_len + victim_len

        # === 桥接文本损失 ===
        # 预测位置: context_len-1 到 context_len+bridge_end_pos-1
        # 目标: target_ids[:, :bridge_end_pos]
        bridge_logits = logits[:, context_len-1:context_len+bridge_end_pos-1, :]
        bridge_targets = target_ids[:, :bridge_end_pos]

        min_bridge_len = min(bridge_logits.shape[1], bridge_targets.shape[1])
        loss_bridge = F.cross_entropy(
            bridge_logits[:, :min_bridge_len, :].reshape(-1, self.vocab_size),
            bridge_targets[:, :min_bridge_len].reshape(-1),
            reduction='mean'
        )

        # === 图像token损失 ===
        # 预测位置: context_len+image_start_pos-1 开始
        # 目标: target_ids[:, image_start_pos:]
        total_target_len = target_ids.shape[1]
        image_len = min(total_target_len - image_start_pos - 1, match_image_len)

        image_logits = logits[:, context_len+image_start_pos-1:context_len+image_start_pos-1+image_len, :]
        image_targets = target_ids[:, image_start_pos:image_start_pos+image_len]

        min_img_len = min(image_logits.shape[1], image_targets.shape[1])
        if min_img_len > 0:
            loss_image = F.cross_entropy(
                image_logits[:, :min_img_len, :].reshape(-1, self.vocab_size),
                image_targets[:, :min_img_len].reshape(-1),
                reduction='mean'
            )
        else:
            loss_image = torch.tensor(0.0, device=self.device)

        # 总损失
        total_loss = lambda_bridge * loss_bridge + lambda_image * loss_image

        loss_dict = {
            'total': total_loss.item(),
            'bridge': loss_bridge.item(),
            'image': loss_image.item(),
        }

        return total_loss, loss_dict

    def optimize(
        self,
        victim_prompt: str,
        bridge_text: str,
        target_image_tokens: List[int],
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        lambda_bridge: float = 2.0,
        lambda_image: float = 1.0,
        match_image_len: int = 256,
        init_strategy: str = "bridge_semantic",
        output_dir: str = "outputs/language_bridge_attack",
    ) -> Tuple[nn.Parameter, Dict]:
        """
        主优化循环
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print("Language Bridge Attack - 语言桥接攻击")
        print(f"{'='*70}")
        print(f"Victim prompt: '{victim_prompt}'")
        print(f"Bridge text: '{bridge_text}'")
        print(f"Target image tokens: {len(target_image_tokens)}")
        print(f"Lambda bridge: {lambda_bridge}, Lambda image: {lambda_image}")
        print(f"{'='*70}\n")

        # 编码
        victim_ids = self.encode_text(victim_prompt)
        target_ids, bridge_end_pos, image_start_pos = self.build_target_sequence(
            bridge_text, target_image_tokens
        )

        print(f"Target sequence length: {target_ids.shape[1]}")
        print(f"Bridge ends at: {bridge_end_pos}, Image starts at: {image_start_pos}")

        # 初始化soft prefix
        soft_prefix = self.init_soft_prefix(
            strategy=init_strategy,
            bridge_text=bridge_text,
        )
        print(f"Soft prefix shape: {soft_prefix.shape}")

        # 优化器
        optimizer = torch.optim.AdamW([soft_prefix], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # 训练
        best_loss = float('inf')
        best_prefix = None
        history = []

        print("\nStarting optimization...")
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()

            loss, loss_dict = self.compute_language_bridge_loss(
                soft_prefix=soft_prefix,
                victim_ids=victim_ids,
                target_ids=target_ids,
                bridge_end_pos=bridge_end_pos,
                image_start_pos=image_start_pos,
                lambda_bridge=lambda_bridge,
                lambda_image=lambda_image,
                match_image_len=match_image_len,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([soft_prefix], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            history.append(loss_dict)

            if loss_dict['total'] < best_loss:
                best_loss = loss_dict['total']
                best_prefix = soft_prefix.detach().clone()

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:4d} | Total: {loss_dict['total']:.4f} | "
                      f"Bridge: {loss_dict['bridge']:.4f} | Image: {loss_dict['image']:.4f}")

            if epoch % 50 == 0:
                torch.cuda.empty_cache()

        # 保存
        final_prefix = best_prefix if best_prefix is not None else soft_prefix.detach()

        checkpoint = {
            'soft_prefix': final_prefix.cpu(),
            'best_loss': best_loss,
            'history': history,
            'config': {
                'victim_prompt': victim_prompt,
                'bridge_text': bridge_text,
                'prefix_length': self.prefix_length,
                'lambda_bridge': lambda_bridge,
                'lambda_image': lambda_image,
                'init_strategy': init_strategy,
                'num_epochs': num_epochs,
            }
        }

        save_path = os.path.join(output_dir, "language_bridge_soft_prompt.pt")
        torch.save(checkpoint, save_path)
        print(f"\n✓ Saved to: {save_path}")

        return final_prefix, {'best_loss': best_loss, 'history': history}

    @torch.no_grad()
    def verify_attack(
        self,
        soft_prefix: torch.Tensor,
        victim_prompt: str,
        max_new_tokens: int = 1200,
    ) -> Tuple[str, List[int], List[int]]:
        """
        验证攻击：检查模型是否输出桥接文本并生成目标图像
        """
        print("\n" + "="*70)
        print("Verifying Language Bridge Attack")
        print("="*70)

        victim_ids = self.encode_text(victim_prompt)
        victim_embeds = self.embedding(victim_ids)

        input_embeds = torch.cat([
            soft_prefix.to(self.model.dtype).to(self.device),
            victim_embeds
        ], dim=1)

        # 自回归生成
        current_embeds = input_embeds
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

        # 解析输出
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

        # 解码文本
        text_output = self.tokenizer.decode(text_tokens, skip_special_tokens=True)

        print(f"Generated text (bridge): '{text_output}'")
        print(f"Generated {len(image_tokens)} image tokens")
        print(f"Total generated: {len(generated_ids)} tokens")

        return text_output, generated_ids, image_tokens

    def harden_with_contrastive(
        self,
        soft_prefix: torch.Tensor,
        positive_text: str,
        negative_text: str,
        top_k: int = 5,
    ) -> Tuple[List[int], List[str], Dict]:
        """
        对比硬化：使离散prompt更像目标（positive），更不像原始（negative）

        Returns:
            token_ids: 硬化后的token IDs
            token_strings: 对应的字符串
            stats: 统计信息
        """
        print("\n" + "="*70)
        print("Contrastive Hardening")
        print("="*70)

        embed_weight = self.embedding.weight  # [vocab_size, hidden]

        # 获取positive和negative的embedding
        pos_ids = self.encode_text(positive_text)
        neg_ids = self.encode_text(negative_text)

        with torch.no_grad():
            pos_emb = self.embedding(pos_ids).mean(dim=1)  # [1, hidden]
            neg_emb = self.embedding(neg_ids).mean(dim=1)  # [1, hidden]

        token_ids = []
        token_strings = []
        scores = []

        soft_prefix = soft_prefix.to(self.device)

        for i in range(soft_prefix.shape[1]):
            prefix_vec = soft_prefix[0, i, :].float()  # [hidden]

            # 计算与词表的相似度
            sims = F.cosine_similarity(
                prefix_vec.unsqueeze(0),
                embed_weight.float(),
                dim=1
            )

            # 对比分数：与positive的相似度 - 与negative的相似度
            pos_bias = F.cosine_similarity(prefix_vec.unsqueeze(0), pos_emb.float(), dim=1)
            neg_bias = F.cosine_similarity(prefix_vec.unsqueeze(0), neg_emb.float(), dim=1)

            # 调整分数
            adjusted_sims = sims + 0.1 * (pos_bias - neg_bias)

            # 排除特殊token
            image_token_ids = self.model.model.vocabulary_mapping.image_token_ids
            for special_id in [self.boi_id, self.eoi_id, 0, 1, 2] + list(image_token_ids):
                if special_id < len(adjusted_sims):
                    adjusted_sims[special_id] = -float('inf')

            # 选择最佳token
            best_id = torch.argmax(adjusted_sims).item()
            token_ids.append(best_id)
            token_strings.append(self.tokenizer.decode([best_id]))
            scores.append(sims[best_id].item())

        # 统计
        stats = {
            'avg_similarity': sum(scores) / len(scores),
            'min_similarity': min(scores),
            'max_similarity': max(scores),
        }

        concatenated = self.tokenizer.decode(token_ids)
        print(f"Hardened prefix: '{concatenated}'")
        print(f"Avg similarity: {stats['avg_similarity']:.4f}")

        return token_ids, token_strings, stats


def main():
    parser = argparse.ArgumentParser(description="Language Bridge Attack")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--target_tokens_file", type=str, required=True)
    parser.add_argument("--victim_prompt", type=str, default="a photo of a cat")
    parser.add_argument("--bridge_text", type=str,
                        default="Sure, I'll generate a cute dog for you.")
    parser.add_argument("--prefix_length", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--lambda_bridge", type=float, default=2.0)
    parser.add_argument("--lambda_image", type=float, default=1.0)
    parser.add_argument("--match_image_len", type=int, default=256)
    parser.add_argument("--init_strategy", type=str, default="bridge_semantic",
                        choices=["bridge_semantic", "target_semantic", "random"])
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(REPO_ROOT, "outputs", "language_bridge_attack"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--harden", action="store_true")

    args = parser.parse_args()

    # 加载目标tokens
    with open(args.target_tokens_file, 'r') as f:
        target_data = json.load(f)
    target_tokens = target_data['image_tokens_list'][0]
    print(f"Loaded {len(target_tokens)} target image tokens")

    # 初始化攻击
    attack = LanguageBridgeAttack(
        model_path=args.model_path,
        prefix_length=args.prefix_length,
    )

    # 优化
    soft_prefix, results = attack.optimize(
        victim_prompt=args.victim_prompt,
        bridge_text=args.bridge_text,
        target_image_tokens=target_tokens,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lambda_bridge=args.lambda_bridge,
        lambda_image=args.lambda_image,
        match_image_len=args.match_image_len,
        init_strategy=args.init_strategy,
        output_dir=args.output_dir,
    )

    # 验证
    if args.verify:
        text_out, full_tokens, image_tokens = attack.verify_attack(
            soft_prefix=soft_prefix.to(attack.device),
            victim_prompt=args.victim_prompt,
        )

        verify_results = {
            'generated_text': text_out,
            'full_tokens': full_tokens,
            'image_tokens': image_tokens,
            'num_image_tokens': len(image_tokens),
        }

        verify_path = os.path.join(args.output_dir, "verification_results.json")
        with open(verify_path, 'w') as f:
            json.dump(verify_results, f, indent=2)
        print(f"Saved verification to: {verify_path}")

    # 对比硬化
    if args.harden:
        token_ids, token_strings, stats = attack.harden_with_contrastive(
            soft_prefix=soft_prefix,
            positive_text="dog puppy canine",
            negative_text="cat kitten feline",
        )

        harden_results = {
            'token_ids': token_ids,
            'token_strings': token_strings,
            'concatenated_text': attack.tokenizer.decode(token_ids),
            'stats': stats,
            'config': {
                'positive_text': "dog puppy canine",
                'negative_text': "cat kitten feline",
            }
        }

        harden_path = os.path.join(args.output_dir, "hardened_prefix.json")
        with open(harden_path, 'w') as f:
            json.dump(harden_results, f, indent=2)
        print(f"Saved hardened prefix to: {harden_path}")


if __name__ == "__main__":
    main()
