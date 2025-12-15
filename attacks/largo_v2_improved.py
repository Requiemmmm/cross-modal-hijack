#!/usr/bin/env python3
"""
LARGO v2: Improved Closed-Loop with Direct Token Generation

Key improvements over v1:
1. Direct multi-token generation (no templates)
2. Token-level nearest neighbor back-projection
3. Better early stopping (based on actual generation)
4. Variable-length prefix (no padding)
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


class LARGOv2:
    """
    LARGO v2: 改进的闭环回投影攻击

    关键改进:
    1. 直接token生成（不使用模板重构）
    2. Token级最近邻回投影
    3. 基于实际生成质量的早停
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        prefix_length: int = 15,  # 减少到15，更灵活
    ):
        self.device = device
        self.prefix_length = prefix_length

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

        # 获取有效token集合
        image_token_ids = set(self.model.model.vocabulary_mapping.image_token_ids)
        special_ids = {self.boi_id, self.eoi_id, 0, 1, 2}
        self.excluded_ids = image_token_ids | special_ids

        print(f"Model loaded. Prefix length: {self.prefix_length}")

    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本为token IDs"""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    def compute_bridge_loss(
        self,
        z: torch.Tensor,
        victim_prompt: str,
        bridge_text: str,
        target_image_tokens: List[int],
        match_image_len: int = 32,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """计算语言桥接损失"""
        victim_ids = self.encode_text(victim_prompt)
        bridge_ids = self.encode_text(bridge_text)

        boi = torch.tensor([[self.boi_id]], device=self.device, dtype=torch.long)
        image_ids = torch.tensor([target_image_tokens[:match_image_len]], device=self.device, dtype=torch.long)

        target_ids = torch.cat([bridge_ids, boi, image_ids], dim=1)

        victim_embeds = self.embedding(victim_ids)

        with torch.no_grad():
            target_embeds = self.embedding(target_ids)

        input_embeds = torch.cat([
            z.to(self.model.dtype),
            victim_embeds,
            target_embeds
        ], dim=1)

        outputs = self.model.model(
            inputs_embeds=input_embeds,
            use_cache=False,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        prefix_len = z.shape[1]
        victim_len = victim_ids.shape[1]
        context_len = prefix_len + victim_len

        # Bridge loss
        bridge_len = bridge_ids.shape[1]
        bridge_logits = logits[:, context_len-1:context_len+bridge_len-1, :]
        bridge_targets = target_ids[:, :bridge_len]

        min_bridge_len = min(bridge_logits.shape[1], bridge_targets.shape[1])
        loss_bridge = F.cross_entropy(
            bridge_logits[:, :min_bridge_len, :].reshape(-1, self.vocab_size),
            bridge_targets[:, :min_bridge_len].reshape(-1),
            reduction='mean'
        )

        # Image prefix loss
        image_start_pos = bridge_len + 1
        image_logits = logits[:, context_len+image_start_pos-1:context_len+image_start_pos-1+match_image_len, :]
        image_targets = target_ids[:, image_start_pos:image_start_pos+match_image_len]

        min_img_len = min(image_logits.shape[1], image_targets.shape[1])
        if min_img_len > 0:
            loss_image = F.cross_entropy(
                image_logits[:, :min_img_len, :].reshape(-1, self.vocab_size),
                image_targets[:, :min_img_len].reshape(-1),
                reduction='mean'
            )
        else:
            loss_image = torch.tensor(0.0, device=self.device)

        total_loss = 2.0 * loss_bridge + 1.0 * loss_image

        loss_dict = {
            'total': total_loss.item(),
            'bridge': loss_bridge.item(),
            'image': loss_image.item(),
        }

        return total_loss, loss_dict, hidden_states

    def gradient_guided_hardening(
        self,
        z: torch.Tensor,
        victim_prompt: str,
        bridge_text: str,
        target_image_tokens: List[int],
        match_image_len: int = 32,
        top_k: int = 64,    # 候选池大小
        batch_size: int = 32, # 批量验证大小
        vocab_constraint: int = 5000,  # 限制在常用词汇
    ) -> List[int]:
        """
        [Critical Upgrade v2] 梯度引导硬化策略 - 修复过时梯度问题

        核心修正：
        - 每个位置重新计算梯度（避免stale gradients）
        - 限制候选池在Top-5000常用词（避免OOD生僻词）
        - 自回归一致性：基于已硬化的前缀计算梯度
        """
        discrete_ids = []
        embed_weight = self.embedding.weight  # [vocab, hidden]

        print(f"    Running Iterative Gradient-Guided Hardening (Top-{top_k}, Batch-{batch_size}, Vocab-{vocab_constraint})...")

        # 逐位置贪婪搜索 - 每步重新计算梯度
        current_z = z.detach().clone()

        for i in range(z.shape[1]):
            # === [关键修正] 基于当前已部分离散化的 current_z 重新计算梯度 ===
            current_z_var = current_z.detach().clone()
            current_z_var.requires_grad = True

            # 1. Forward & Backward（获取fresh gradient）
            loss, _, _ = self.compute_bridge_loss(
                current_z_var, victim_prompt, bridge_text, target_image_tokens, match_image_len
            )
            loss.backward()
            fresh_grad = current_z_var.grad  # [1, prefix_len, hidden]

            # --- Step A: 筛选候选 (Filtering via HotFlip with Fresh Gradient) ---
            # HotFlip Score = - (Token_Embed @ Fresh_Gradient)

            # [关键修正] 限制候选池为Top-5000常用词，防止生成医学/生僻词
            common_vocab_embeds = embed_weight[:vocab_constraint].float()
            token_scores = torch.matmul(common_vocab_embeds, -fresh_grad[0, i, :].float())

            # 排除特殊 Token（在常用词范围内）
            for special_id in self.excluded_ids:
                if special_id < len(token_scores):
                    token_scores[special_id] = -float('inf')

            # 获取 top-K 候选（索引已经限制在0-vocab_constraint范围内）
            top_k_limited = min(top_k, len(token_scores))
            top_indices = torch.topk(token_scores, top_k_limited).indices.tolist()

            # 确保原来的几何最近邻也在候选池里（保底）- 也限制在常用词
            with torch.no_grad():
                sim_scores = F.cosine_similarity(
                    current_z[0, i, :].unsqueeze(0),
                    common_vocab_embeds,
                    dim=1
                )
                for special_id in self.excluded_ids:
                    if special_id < len(sim_scores):
                        sim_scores[special_id] = -float('inf')
                original_nearest = torch.argmax(sim_scores).item()

            if original_nearest not in top_indices:
                top_indices.append(original_nearest)

            # --- Step B: 真实评估 (Evaluation) ---
            best_token_loss = float('inf')
            best_token_id = top_indices[0]

            # 对候选进行验证
            candidates_to_test = top_indices[:batch_size]

            for cand_id in candidates_to_test:
                # 构造探测向量：只修改当前位置 i
                z_probe = current_z.clone()
                cand_emb = self.embedding(torch.tensor([[cand_id]], device=self.device))
                z_probe[0, i, :] = cand_emb.to(z_probe.dtype)

                with torch.no_grad():
                    l, _, _ = self.compute_bridge_loss(
                        z_probe, victim_prompt, bridge_text, target_image_tokens, match_image_len
                    )

                if l.item() < best_token_loss:
                    best_token_loss = l.item()
                    best_token_id = cand_id

            # 锁定当前位置的最佳 Token
            discrete_ids.append(best_token_id)

            # 更新 current_z，让下一个位置的搜索基于当前的硬化结果 (Greedy)
            # 这保持了序列生成的自回归一致性
            with torch.no_grad():
                best_emb = self.embedding(torch.tensor([[best_token_id]], device=self.device))
                current_z[0, i, :] = best_emb.to(current_z.dtype)

            if (i + 1) % 5 == 0 or i == 0:
                decoded_token = self.tokenizer.decode([best_token_id])
                print(f"      Position {i+1}/{z.shape[1]}: '{decoded_token}' | Loss={best_token_loss:.4f}")

        return discrete_ids

    @torch.no_grad()
    def test_discrete_prefix(
        self,
        discrete_ids: List[int],
        victim_prompt: str,
        max_new_tokens: int = 150,  # 只生成前150个token快速测试
    ) -> Dict:
        """
        快速测试离散prefix是否能生成有效输出

        返回:
        - generated_text: 生成的文本
        - has_image: 是否生成了图像
        - is_blank: 图像是否全黑/全白（判断失败）
        """
        prefix_tensor = torch.tensor([discrete_ids], device=self.device, dtype=torch.long)
        victim_ids = self.encode_text(victim_prompt)

        input_ids = torch.cat([prefix_tensor, victim_ids], dim=1)
        current_embeds = self.embedding(input_ids).to(self.model.dtype)

        generated_ids = []
        found_boi = False
        image_token_count = 0

        for step in range(max_new_tokens):
            outputs = self.model.model(
                inputs_embeds=current_embeds,
                use_cache=False,
            )

            logits = self.lm_head(outputs.last_hidden_state[:, -1:, :])
            next_id = torch.argmax(logits, dim=-1).item()
            generated_ids.append(next_id)

            if next_id == self.boi_id:
                found_boi = True
            elif next_id == self.eoi_id:
                break
            elif found_boi:
                image_token_count += 1

            if image_token_count >= 32:  # 生成了至少32个图像token，足够判断
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

        text_output = self.tokenizer.decode(text_tokens, skip_special_tokens=True)

        # 检查是否是blank image（同一token重复）
        is_blank = False
        if len(image_tokens) >= 20:
            # 检查前20个token是否大部分相同
            unique_ratio = len(set(image_tokens[:20])) / 20
            is_blank = unique_ratio < 0.3  # 如果唯一token<30%，认为是blank

        result = {
            'text': text_output,
            'has_image': len(image_tokens) > 0,
            'image_token_count': len(image_tokens),
            'is_blank': is_blank,
            'quality_score': 1.0 if (len(image_tokens) > 0 and not is_blank) else 0.0,
        }

        return result

    def largo_iteration(
        self,
        z: torch.Tensor,
        victim_prompt: str,
        bridge_text: str,
        target_image_tokens: List[int],
        reference_hidden: Optional[torch.Tensor] = None,
        latent_steps: int = 50,
        learning_rate: float = 0.01,
        anchor_reg: float = 0.05,
        compact_reg: float = 0.2,  # 离散化引力权重
        match_image_len: int = 32,
        top_k: int = 64,
        batch_size: int = 32,
    ) -> Tuple[torch.Tensor, List[int], Dict]:
        """LARGO单次迭代 - v3最终版本（梯度引导+离散化引力）"""

        # === Stage 1: Latent Optimization with Discreteness Regularization ===
        z_start = z.detach().clone()
        z_opt = z.clone()
        z_opt.requires_grad = True

        optimizer = torch.optim.Adam([z_opt], lr=learning_rate)

        best_loss = float('inf')
        best_z = None

        print(f"    Stage 1: Continuous Optimization (with Discreteness Gravity)")

        for step in range(latent_steps):
            optimizer.zero_grad()

            loss_task, loss_dict, _ = self.compute_bridge_loss(
                z_opt, victim_prompt, bridge_text, target_image_tokens, match_image_len
            )

            loss_anchor = torch.norm(z_opt - z_start)

            # === 新增：Discreteness Loss (离散化引力) ===
            # 强迫 z 靠近最近的有效 embedding
            with torch.no_grad():
                # 找到当前几何上最近的 embeddings (Hard Snap Targets)
                # [1, prefix_len, hidden] vs [vocab, hidden]
                z_opt_expanded = z_opt.unsqueeze(2)  # [1, prefix_len, 1, hidden]
                embed_expanded = self.embedding.weight.unsqueeze(0).unsqueeze(0)  # [1, 1, vocab, hidden]

                # 使用更高效的计算方式
                sims = []
                for i in range(z_opt.shape[1]):
                    sim = F.cosine_similarity(
                        z_opt[0, i, :].unsqueeze(0),
                        self.embedding.weight.float(),
                        dim=1
                    )
                    # 排除特殊token
                    for special_id in self.excluded_ids:
                        if special_id < len(sim):
                            sim[special_id] = -float('inf')
                    nearest_id = torch.argmax(sim)
                    sims.append(nearest_id)

                nearest_ids = torch.stack(sims).unsqueeze(0)  # [1, prefix_len]
                nearest_embeds = self.embedding(nearest_ids).detach()

            # 计算 MSE 引力
            loss_compact = F.mse_loss(z_opt, nearest_embeds.to(z_opt.dtype))

            # 组合 Loss: Task为主，Anchor防漂移，Compact促离散
            loss = loss_task + anchor_reg * loss_anchor + compact_reg * loss_compact

            loss.backward()
            torch.nn.utils.clip_grad_norm_([z_opt], max_norm=1.0)
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z_opt.detach().clone()

            if step % 10 == 0:
                print(f"      Step {step:2d} | Task: {loss_dict['total']:.4f} | "
                      f"Anchor: {loss_anchor.item():.4f} | Compact: {loss_compact.item():.4f}")

        z_optimized = best_z if best_z is not None else z_opt.detach()

        # === Stage 2: Gradient-Guided Hardening (核心升级) ===
        print(f"    Stage 2: Gradient-Guided Hardening")
        discrete_ids = self.gradient_guided_hardening(
            z_optimized,
            victim_prompt,
            bridge_text,
            target_image_tokens,
            match_image_len,
            top_k,
            batch_size,
            vocab_constraint=5000,  # 限制在常用词
        )
        discrete_text = self.tokenizer.decode(discrete_ids)

        print(f"    Hardened prefix ({len(discrete_ids)} tokens): '{discrete_text}'")

        # === Stage 3: Quality Test ===
        test_result = self.test_discrete_prefix(discrete_ids, victim_prompt)
        print(f"    Test result: {test_result['text'][:50]}... | "
              f"Image: {test_result['has_image']} | Blank: {test_result['is_blank']} | "
              f"Quality: {test_result['quality_score']:.2f}")

        # === Stage 4: Back-Projection ===
        with torch.no_grad():
            discrete_tensor = torch.tensor([discrete_ids], device=self.device, dtype=torch.long)
            z_new = self.embedding(discrete_tensor).float()

        iteration_stats = {
            'best_loss': best_loss,
            'discrete_ids': discrete_ids,
            'discrete_text': discrete_text,
            'test_result': test_result,
            'quality_score': test_result['quality_score'],
        }

        return z_new, discrete_ids, iteration_stats

    def run_largo_loop(
        self,
        victim_prompt: str,
        bridge_text: str,
        target_image_tokens: List[int],
        initial_soft_prompt: Optional[torch.Tensor] = None,
        max_iterations: int = 3,
        latent_steps: int = 50,
        learning_rate: float = 0.01,
        anchor_reg: float = 0.05,
        compact_reg: float = 0.2,
        match_image_len: int = 32,
        top_k: int = 64,
        batch_size: int = 32,
        output_dir: str = "outputs/largo_v2",
    ) -> Tuple[List[int], str, List[Dict]]:
        """运行LARGO v2闭环迭代"""

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print("LARGO v2: Improved Closed-Loop Attack")
        print(f"{'='*70}")
        print(f"Victim prompt: '{victim_prompt}'")
        print(f"Bridge text: '{bridge_text}'")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*70}\n")

        # 初始化z
        if initial_soft_prompt is not None:
            z = initial_soft_prompt.to(self.device).float()
            # 调整长度
            if z.shape[1] > self.prefix_length:
                z = z[:, :self.prefix_length, :]
            elif z.shape[1] < self.prefix_length:
                padding = torch.zeros(1, self.prefix_length - z.shape[1], self.hidden_size, device=self.device)
                z = torch.cat([z, padding], dim=1)
            print(f"Using provided soft prompt (adjusted to {self.prefix_length} tokens)")
        else:
            bridge_ids = self.encode_text(bridge_text)
            with torch.no_grad():
                bridge_emb = self.embedding(bridge_ids)
                mean_emb = bridge_emb.mean(dim=1, keepdim=True)
                z = mean_emb.repeat(1, self.prefix_length, 1).float()
                z = z + torch.randn_like(z) * 0.01
            print(f"Initialized z from bridge text")

        print(f"z shape: {z.shape}\n")

        # 迭代
        history = []
        best_overall_quality = 0.0
        best_discrete_ids = None
        best_discrete_text = None

        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"LARGO v2 Iteration {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")

            z_new, discrete_ids, iter_stats = self.largo_iteration(
                z=z,
                victim_prompt=victim_prompt,
                bridge_text=bridge_text,
                target_image_tokens=target_image_tokens,
                latent_steps=latent_steps,
                learning_rate=learning_rate,
                anchor_reg=anchor_reg,
                compact_reg=compact_reg,
                match_image_len=match_image_len,
                top_k=top_k,
                batch_size=batch_size,
            )

            iter_stats['iteration'] = iteration + 1
            history.append(iter_stats)

            print(f"\n  Iteration {iteration + 1} Summary:")
            print(f"    Loss: {iter_stats['best_loss']:.4f}")
            print(f"    Quality: {iter_stats['quality_score']:.2f}")
            print(f"    Discrete text: '{iter_stats['discrete_text'][:60]}...'")

            # 更新最佳（基于质量而非loss）
            if iter_stats['quality_score'] > best_overall_quality:
                best_overall_quality = iter_stats['quality_score']
                best_discrete_ids = discrete_ids
                best_discrete_text = iter_stats['discrete_text']
                print(f"    ✓ New best! (quality: {best_overall_quality:.2f})")

            # 早停：如果生成质量达标
            if iter_stats['quality_score'] >= 0.8:
                print(f"\n  Quality threshold reached, stopping early.")
                break

            # 回投影
            z = z_new

        print(f"\n{'='*70}")
        print("LARGO v2 Loop Complete")
        print(f"{'='*70}")
        print(f"Best quality: {best_overall_quality:.2f}")
        print(f"Best discrete prefix: '{best_discrete_text[:80]}...'")
        print(f"{'='*70}\n")

        # 保存结果
        results = {
            'best_discrete_ids': best_discrete_ids,
            'best_discrete_text': best_discrete_text,
            'best_quality': best_overall_quality,
            'history': history,
            'config': {
                'victim_prompt': victim_prompt,
                'bridge_text': bridge_text,
                'max_iterations': max_iterations,
                'latent_steps': latent_steps,
                'learning_rate': learning_rate,
                'anchor_reg': anchor_reg,
                'compact_reg': compact_reg,
                'match_image_len': match_image_len,
                'top_k': top_k,
                'batch_size': batch_size,
            }
        }

        results_path = os.path.join(output_dir, "largo_v2_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {results_path}")

        return best_discrete_ids, best_discrete_text, history

    @torch.no_grad()
    def verify_full_generation(
        self,
        discrete_ids: List[int],
        victim_prompt: str,
        max_new_tokens: int = 1200,
    ) -> Tuple[str, List[int], List[int]]:
        """完整生成验证"""
        print("\n" + "="*70)
        print("Full Generation Verification")
        print("="*70)

        prefix_tensor = torch.tensor([discrete_ids], device=self.device, dtype=torch.long)
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
    parser = argparse.ArgumentParser(description="LARGO v2 Attack")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--target_tokens_file", type=str, required=True)
    parser.add_argument("--victim_prompt", type=str, default="a photo of a cat")
    parser.add_argument("--bridge_text", type=str,
                        default="Sure, I'll generate a cute dog for you.")
    parser.add_argument("--initial_soft_prompt", type=str, default=None)
    parser.add_argument("--prefix_length", type=int, default=15)
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--latent_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--anchor_reg", type=float, default=0.05)
    parser.add_argument("--compact_reg", type=float, default=0.2,
                        help="Discreteness regularization weight")
    parser.add_argument("--match_image_len", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=64,
                        help="Top-K candidates for gradient-guided hardening")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for candidate evaluation")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(REPO_ROOT, "outputs", "largo_v3"))
    parser.add_argument("--verify", action="store_true")

    args = parser.parse_args()

    # 加载目标tokens
    with open(args.target_tokens_file, 'r') as f:
        target_data = json.load(f)
    target_tokens = target_data['image_tokens_list'][0]
    print(f"Loaded {len(target_tokens)} target image tokens")

    # 加载初始soft prompt
    initial_soft = None
    if args.initial_soft_prompt and os.path.exists(args.initial_soft_prompt):
        print(f"Loading initial soft prompt from: {args.initial_soft_prompt}")
        checkpoint = torch.load(args.initial_soft_prompt, map_location="cpu", weights_only=False)
        initial_soft = checkpoint['soft_prefix']
        print(f"Initial soft prompt shape: {initial_soft.shape}")

    # 初始化
    largo = LARGOv2(
        model_path=args.model_path,
        prefix_length=args.prefix_length,
    )

    # 运行LARGO v3
    best_ids, best_text, history = largo.run_largo_loop(
        victim_prompt=args.victim_prompt,
        bridge_text=args.bridge_text,
        target_image_tokens=target_tokens,
        initial_soft_prompt=initial_soft,
        max_iterations=args.max_iterations,
        latent_steps=args.latent_steps,
        learning_rate=args.learning_rate,
        anchor_reg=args.anchor_reg,
        compact_reg=args.compact_reg,
        match_image_len=args.match_image_len,
        top_k=args.top_k,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    # 完整验证
    if args.verify:
        text_out, full_tokens, image_tokens = largo.verify_full_generation(
            best_ids,
            args.victim_prompt,
        )

        verify_results = {
            'prefix_ids': best_ids,
            'prefix_text': best_text,
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
