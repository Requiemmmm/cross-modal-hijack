#!/usr/bin/env python3
"""
LARGO Closed-Loop Back-Projection Attack (LARGO 闭环回投影攻击)

核心改进（基于用户建议）：
1. **Back-Projection 闭环**: Soft → Discrete → Re-embed → Optimize (迭代)
2. **Language Bridge 专注**: 全力攻击文本桥接，利用近因效应劫持图像生成
3. **Hidden State Matching**: 利用白盒成功的隐藏状态作为锚点
4. **Teacher Forcing优化**: 只优化前16-32个图像token，提升效率100倍

方法对比：
- 旧方法（开环）: Soft → Nearest Neighbor → Fail (信息损失90%)
- LARGO（闭环）: Soft → Discrete → Re-embed → Optimize → ... → Success

关键洞察：
- Language Bridge在白盒下极其有效（Loss: 15.56 → 0.005）
- 文本生成比图像生成鲁棒得多
- 一旦模型输出"Sure, I'll generate a cute dog"，近因效应会压倒原始"cat" prompt
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


class LARGOClosedLoop:
    """
    LARGO闭环回投影攻击

    核心流程：
    1. Latent Optimization: 优化连续embedding（加正则化）
    2. Reflective Decoding: 补全式模板诱导生成目标词
    3. Back-Projection: 将生成的离散文本重新embed，作为下一轮起点

    重复迭代直到收敛或达到最大轮数
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        prefix_length: int = 20,
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

        # Freeze model
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

        print(f"Model loaded. Hidden size: {self.hidden_size}")
        print(f"BOI: {self.boi_id}, EOI: {self.eoi_id}")

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
        match_image_len: int = 32,  # 只匹配前32个token（Teacher Forcing）
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算语言桥接损失 + 图像前缀损失（Teacher Forcing）

        Args:
            z: 连续embedding [1, prefix_len, hidden]
            victim_prompt: 受害者prompt（如"a photo of a cat"）
            bridge_text: 桥接文本（如"Sure, I'll generate a cute dog for you."）
            target_image_tokens: 目标图像的1024个token
            match_image_len: 只匹配前N个图像token（效率优化）
        """
        # 编码victim和target
        victim_ids = self.encode_text(victim_prompt)
        bridge_ids = self.encode_text(bridge_text)

        boi = torch.tensor([[self.boi_id]], device=self.device, dtype=torch.long)
        image_ids = torch.tensor([target_image_tokens[:match_image_len]], device=self.device, dtype=torch.long)

        # 目标序列: [bridge] + [BOI] + [image_prefix]
        target_ids = torch.cat([bridge_ids, boi, image_ids], dim=1)

        # 获取embeddings
        victim_embeds = self.embedding(victim_ids)

        with torch.no_grad():
            target_embeds = self.embedding(target_ids)

        # 输入: [z] + [victim] + [target]
        input_embeds = torch.cat([
            z.to(self.model.dtype),
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
        prefix_len = z.shape[1]
        victim_len = victim_ids.shape[1]
        context_len = prefix_len + victim_len

        # Bridge loss（最重要）
        bridge_len = bridge_ids.shape[1]
        bridge_logits = logits[:, context_len-1:context_len+bridge_len-1, :]
        bridge_targets = target_ids[:, :bridge_len]

        min_bridge_len = min(bridge_logits.shape[1], bridge_targets.shape[1])
        loss_bridge = F.cross_entropy(
            bridge_logits[:, :min_bridge_len, :].reshape(-1, self.vocab_size),
            bridge_targets[:, :min_bridge_len].reshape(-1),
            reduction='mean'
        )

        # Image prefix loss（用Teacher Forcing，只计算前N个token）
        image_start_pos = bridge_len + 1  # +1 for BOI
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

        # 总损失（强调bridge）
        total_loss = 2.0 * loss_bridge + 1.0 * loss_image

        loss_dict = {
            'total': total_loss.item(),
            'bridge': loss_bridge.item(),
            'image': loss_image.item(),
        }

        return total_loss, loss_dict, hidden_states

    def compute_hidden_state_loss(
        self,
        current_hidden: torch.Tensor,
        reference_hidden: torch.Tensor,
        prefix_len: int,
        victim_len: int,
    ) -> torch.Tensor:
        """
        Hidden State Matching Loss

        利用白盒成功的隐藏状态作为锚点，缩小搜索空间

        Args:
            current_hidden: 当前的隐藏状态 [1, seq_len, hidden]
            reference_hidden: 白盒成功的参考隐藏状态 [1, ref_len, hidden]
            prefix_len: prefix长度
            victim_len: victim prompt长度
        """
        # 只匹配context部分（prefix + victim）的隐藏状态
        context_len = prefix_len + victim_len

        if reference_hidden is None:
            return torch.tensor(0.0, device=self.device)

        # 确保长度匹配
        min_len = min(context_len, reference_hidden.shape[1])

        current_context = current_hidden[:, :min_len, :]
        reference_context = reference_hidden[:, :min_len, :].to(self.device)

        # L2距离
        loss_hidden = F.mse_loss(current_context, reference_context)

        return loss_hidden

    @torch.no_grad()
    def reflective_decode_completion(
        self,
        z: torch.Tensor,
        max_new_tokens: int = 3,
        temperature: float = 0.7,
    ) -> Tuple[str, List[int]]:
        """
        补全式反射解码（关键改进）

        旧方法: "Summarize: [z] → The instruction is about..."（模型进入推理模式）
        新方法: "User: [z]\nAssistant: Sure, here is a generated image of a"（诱导补全）

        这利用了Language Bridge的成功经验，诱导模型说出目标词（如"dog"）
        """
        # 构建补全式模板
        user_prefix = self.encode_text("User: ")
        assistant_prefix = self.encode_text("\nAssistant: Sure, here is a generated image of a")

        user_emb = self.embedding(user_prefix)
        assistant_emb = self.embedding(assistant_prefix)

        # [User: ] + [z] + [\nAssistant: Sure, here is a generated image of a]
        input_embeds = torch.cat([
            user_emb.to(self.model.dtype),
            z.to(self.model.dtype),
            assistant_emb.to(self.model.dtype)
        ], dim=1)

        current_embeds = input_embeds
        generated_ids = []

        # 只生成1-3个token（限制模型不进入推理模式）
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

            # Stop at special tokens or end of sentence
            if next_id in [self.boi_id, self.eoi_id, self.tokenizer.eos_token_id]:
                break

            next_emb = self.embedding(torch.tensor([[next_id]], device=self.device))
            current_embeds = torch.cat([current_embeds, next_emb.to(self.model.dtype)], dim=1)

        decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return decoded_text, generated_ids

    def reconstruct_prefix_text(
        self,
        reflected_word: str,
        template: str = "Generate an image of a {word}."
    ) -> str:
        """
        从反射出的词重构完整的prefix文本

        例如: reflected_word = "dog" → "Generate an image of a dog."
        """
        return template.format(word=reflected_word)

    def largo_iteration(
        self,
        z: torch.Tensor,
        victim_prompt: str,
        bridge_text: str,
        target_image_tokens: List[int],
        reference_hidden: Optional[torch.Tensor] = None,
        latent_steps: int = 50,
        learning_rate: float = 0.01,
        anchor_reg: float = 0.1,
        hidden_reg: float = 0.5,
        match_image_len: int = 32,
    ) -> Tuple[torch.Tensor, List[int], str, Dict]:
        """
        LARGO单次迭代

        流程:
        1. Latent Optimization: 优化z（带锚点正则化和隐藏状态匹配）
        2. Reflective Decoding: 补全式解码得到目标词
        3. Reconstruction: 重构完整prefix文本
        4. Return: 新的离散token和embedding
        """
        # === Stage 1: Latent Optimization ===
        z_start = z.detach().clone()
        z_opt = z.clone()
        z_opt.requires_grad = True

        optimizer = torch.optim.Adam([z_opt], lr=learning_rate)

        best_loss = float('inf')
        best_z = None

        for step in range(latent_steps):
            optimizer.zero_grad()

            # 任务损失
            loss_task, loss_dict, hidden_states = self.compute_bridge_loss(
                z_opt, victim_prompt, bridge_text, target_image_tokens, match_image_len
            )

            # 锚点正则化（防止z跑太远）
            loss_anchor = torch.norm(z_opt - z_start)

            # 隐藏状态匹配损失（利用白盒参考）
            if reference_hidden is not None:
                victim_len = len(self.encode_text(victim_prompt)[0])
                loss_hidden = self.compute_hidden_state_loss(
                    hidden_states, reference_hidden, self.prefix_length, victim_len
                )
            else:
                loss_hidden = torch.tensor(0.0, device=self.device)

            # 总损失
            loss = loss_task + anchor_reg * loss_anchor + hidden_reg * loss_hidden

            loss.backward()
            torch.nn.utils.clip_grad_norm_([z_opt], max_norm=1.0)
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z_opt.detach().clone()

            if step % 10 == 0:
                print(f"    Latent step {step:2d} | Task: {loss_dict['total']:.4f} | "
                      f"Anchor: {loss_anchor.item():.4f} | Hidden: {loss_hidden.item():.4f}")

        z_optimized = best_z if best_z is not None else z_opt.detach()

        # === Stage 2: Reflective Decoding ===
        reflected_word, reflected_ids = self.reflective_decode_completion(z_optimized)

        print(f"    Reflected word: '{reflected_word}'")

        # === Stage 3: Reconstruction ===
        # 重构完整prefix（可以是简单的模板，也可以更复杂）
        full_prefix_text = self.reconstruct_prefix_text(reflected_word)
        full_prefix_ids = self.encode_text(full_prefix_text)[0].tolist()

        print(f"    Reconstructed prefix: '{full_prefix_text}'")

        # === Stage 4: Back-Projection（关键步骤）===
        # 将离散文本重新embed，作为下一轮的起点
        # 这消除了离散化带来的信息损失
        with torch.no_grad():
            # 如果重构的prefix长度不等于原始长度，需要调整
            if len(full_prefix_ids) < self.prefix_length:
                # Pad
                full_prefix_ids = full_prefix_ids + [self.tokenizer.pad_token_id] * (self.prefix_length - len(full_prefix_ids))
            elif len(full_prefix_ids) > self.prefix_length:
                # Truncate
                full_prefix_ids = full_prefix_ids[:self.prefix_length]

            new_discrete_tensor = torch.tensor([full_prefix_ids], device=self.device, dtype=torch.long)
            z_new = self.embedding(new_discrete_tensor).float()

        iteration_stats = {
            'best_loss': best_loss,
            'reflected_word': reflected_word,
            'reconstructed_text': full_prefix_text,
            'discrete_ids': full_prefix_ids,
        }

        return z_new, full_prefix_ids, full_prefix_text, iteration_stats

    def run_largo_loop(
        self,
        victim_prompt: str,
        bridge_text: str,
        target_image_tokens: List[int],
        initial_soft_prompt: Optional[torch.Tensor] = None,
        reference_hidden_path: Optional[str] = None,
        max_iterations: int = 5,
        latent_steps: int = 50,
        learning_rate: float = 0.01,
        anchor_reg: float = 0.1,
        hidden_reg: float = 0.5,
        match_image_len: int = 32,
        output_dir: str = "outputs/largo_closed_loop",
    ) -> Tuple[List[int], str, List[Dict]]:
        """
        运行完整的LARGO闭环迭代

        流程:
        初始化 → 迭代N次 → 返回最佳离散prefix
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print("LARGO Closed-Loop Back-Projection Attack")
        print(f"{'='*70}")
        print(f"Victim prompt: '{victim_prompt}'")
        print(f"Bridge text: '{bridge_text}'")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*70}\n")

        # 加载白盒参考隐藏状态（如果提供）
        reference_hidden = None
        if reference_hidden_path and os.path.exists(reference_hidden_path):
            print(f"Loading reference hidden states from: {reference_hidden_path}")
            ref_checkpoint = torch.load(reference_hidden_path, map_location="cpu")
            if 'hidden_states' in ref_checkpoint:
                reference_hidden = ref_checkpoint['hidden_states']
                print(f"Reference hidden states shape: {reference_hidden.shape}")
            elif 'soft_prefix' in ref_checkpoint:
                # 如果只有soft_prefix，需要先forward一次得到hidden states
                print("Reference file contains soft_prefix, computing hidden states...")
                soft_prefix_ref = ref_checkpoint['soft_prefix'].to(self.device)
                victim_ids = self.encode_text(victim_prompt)
                victim_embeds = self.embedding(victim_ids)
                input_embeds = torch.cat([soft_prefix_ref.to(self.model.dtype), victim_embeds], dim=1)

                with torch.no_grad():
                    outputs = self.model.model(inputs_embeds=input_embeds, use_cache=False)
                    reference_hidden = outputs.last_hidden_state.cpu()
                print(f"Computed reference hidden states shape: {reference_hidden.shape}")

        # 初始化z
        if initial_soft_prompt is not None:
            z = initial_soft_prompt.to(self.device).float()
            print(f"Using provided soft prompt as initialization")
        else:
            # 从bridge text的语义初始化
            bridge_ids = self.encode_text(bridge_text)
            with torch.no_grad():
                bridge_emb = self.embedding(bridge_ids)
                mean_emb = bridge_emb.mean(dim=1, keepdim=True)
                z = mean_emb.repeat(1, self.prefix_length, 1).float()
                z = z + torch.randn_like(z) * 0.01
            print(f"Initialized z from bridge text semantics")

        print(f"z shape: {z.shape}\n")

        # 迭代
        history = []
        best_overall_loss = float('inf')
        best_discrete_ids = None
        best_discrete_text = None

        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"LARGO Iteration {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")

            z_new, discrete_ids, discrete_text, iter_stats = self.largo_iteration(
                z=z,
                victim_prompt=victim_prompt,
                bridge_text=bridge_text,
                target_image_tokens=target_image_tokens,
                reference_hidden=reference_hidden,
                latent_steps=latent_steps,
                learning_rate=learning_rate,
                anchor_reg=anchor_reg,
                hidden_reg=hidden_reg,
                match_image_len=match_image_len,
            )

            iter_stats['iteration'] = iteration + 1
            history.append(iter_stats)

            print(f"\n  Iteration {iteration + 1} Summary:")
            print(f"    Loss: {iter_stats['best_loss']:.4f}")
            print(f"    Discrete text: '{discrete_text}'")

            # 更新最佳
            if iter_stats['best_loss'] < best_overall_loss:
                best_overall_loss = iter_stats['best_loss']
                best_discrete_ids = discrete_ids
                best_discrete_text = discrete_text
                print(f"    ✓ New best!")

            # 回投影: 用新的embedding作为下一轮起点
            z = z_new

            # 早停：如果loss足够低
            if iter_stats['best_loss'] < 0.5:
                print(f"\n  Loss threshold reached, stopping early.")
                break

        print(f"\n{'='*70}")
        print("LARGO Loop Complete")
        print(f"{'='*70}")
        print(f"Best loss: {best_overall_loss:.4f}")
        print(f"Best discrete prefix: '{best_discrete_text}'")
        print(f"{'='*70}\n")

        # 保存结果
        results = {
            'best_discrete_ids': best_discrete_ids,
            'best_discrete_text': best_discrete_text,
            'best_loss': best_overall_loss,
            'history': history,
            'config': {
                'victim_prompt': victim_prompt,
                'bridge_text': bridge_text,
                'max_iterations': max_iterations,
                'latent_steps': latent_steps,
                'learning_rate': learning_rate,
                'anchor_reg': anchor_reg,
                'hidden_reg': hidden_reg,
                'match_image_len': match_image_len,
            }
        }

        results_path = os.path.join(output_dir, "largo_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {results_path}")

        return best_discrete_ids, best_discrete_text, history

    @torch.no_grad()
    def verify_discrete_attack(
        self,
        discrete_ids: List[int],
        victim_prompt: str,
        max_new_tokens: int = 1200,
    ) -> Tuple[str, List[int], List[int]]:
        """
        验证离散prefix的黑盒攻击效果
        """
        print("\n" + "="*70)
        print("Verifying Discrete Prefix Attack")
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

        print(f"Generated text: '{text_output}'")
        print(f"Generated {len(image_tokens)} image tokens")
        print("="*70)

        return text_output, generated_ids, image_tokens


def main():
    parser = argparse.ArgumentParser(description="LARGO Closed-Loop Attack")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--target_tokens_file", type=str, required=True)
    parser.add_argument("--victim_prompt", type=str, default="a photo of a cat")
    parser.add_argument("--bridge_text", type=str,
                        default="Sure, I'll generate a cute dog for you.")
    parser.add_argument("--initial_soft_prompt", type=str, default=None,
                        help="Path to initial soft prompt (.pt file)")
    parser.add_argument("--reference_hidden", type=str, default=None,
                        help="Path to white-box reference hidden states")
    parser.add_argument("--prefix_length", type=int, default=20)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--latent_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--anchor_reg", type=float, default=0.1)
    parser.add_argument("--hidden_reg", type=float, default=0.5)
    parser.add_argument("--match_image_len", type=int, default=32)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(REPO_ROOT, "outputs", "largo_closed_loop"))
    parser.add_argument("--verify", action="store_true")

    args = parser.parse_args()

    # 加载目标tokens
    with open(args.target_tokens_file, 'r') as f:
        target_data = json.load(f)
    target_tokens = target_data['image_tokens_list'][0]
    print(f"Loaded {len(target_tokens)} target image tokens")

    # 加载初始soft prompt（如果提供）
    initial_soft = None
    if args.initial_soft_prompt and os.path.exists(args.initial_soft_prompt):
        print(f"Loading initial soft prompt from: {args.initial_soft_prompt}")
        checkpoint = torch.load(args.initial_soft_prompt, map_location="cpu")
        initial_soft = checkpoint['soft_prefix']
        print(f"Initial soft prompt shape: {initial_soft.shape}")

    # 初始化攻击
    largo = LARGOClosedLoop(
        model_path=args.model_path,
        prefix_length=args.prefix_length,
    )

    # 运行LARGO闭环
    best_ids, best_text, history = largo.run_largo_loop(
        victim_prompt=args.victim_prompt,
        bridge_text=args.bridge_text,
        target_image_tokens=target_tokens,
        initial_soft_prompt=initial_soft,
        reference_hidden_path=args.reference_hidden,
        max_iterations=args.max_iterations,
        latent_steps=args.latent_steps,
        learning_rate=args.learning_rate,
        anchor_reg=args.anchor_reg,
        hidden_reg=args.hidden_reg,
        match_image_len=args.match_image_len,
        output_dir=args.output_dir,
    )

    # 验证
    if args.verify:
        text_out, full_tokens, image_tokens = largo.verify_discrete_attack(
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
