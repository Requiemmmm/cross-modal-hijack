#!/usr/bin/env python3
"""
生成目标对象的图像 tokens
用于 Phase 2 的对抗性优化目标
"""
import os
import sys
import json
import torch
import argparse
from typing import List

# Add local transformers to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
local_tf_src = os.path.join(repo_root, "transformers", "src")
if os.path.isdir(local_tf_src) and local_tf_src not in sys.path:
    sys.path.insert(0, local_tf_src)

from transformers import ChameleonConfig, ChameleonProcessor, ChameleonForConditionalGenerationWithCFG


def generate_target_image_tokens(
    model_path: str,
    target_prompt: str,
    num_samples: int = 5,
    device: str = "cuda",
    output_file: str = None
):
    """
    为目标对象生成多个参考图像的 tokens

    Args:
        model_path: 模型路径
        target_prompt: 目标提示词（例如 "a photo of a dog"）
        num_samples: 生成样本数量
        device: 设备
        output_file: 输出文件路径
    """
    print(f"Loading model from {model_path}...")

    # Load model and processor
    config = ChameleonConfig.from_pretrained(model_path)
    config._attn_implementation = "flash_attention_2" if device == "cuda" else "eager"

    processor = ChameleonProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = ChameleonForConditionalGenerationWithCFG.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    # Disable CFG for simplicity
    model.setup_cfg(cfg_config="no")

    boi_token_id = config.boi_token_id
    eoi_token_id = config.eoi_token_id

    print(f"Generating {num_samples} samples for: '{target_prompt}'")

    all_image_tokens = []

    for i in range(num_samples):
        print(f"\nGenerating sample {i+1}/{num_samples}...")

        # Prepare input
        inputs = processor(
            target_prompt,
            padding=False,
            return_tensors="pt",
            return_for_text_completion=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Generate with greedy decoding for consistency
            outputs = model.generate(
                **inputs,
                max_new_tokens=1100,
                do_sample=False,  # Greedy for first sample, then sample
                temperature=0.9 if i > 0 else 1.0,
                multimodal_generation_mode="interleaved-text-image",
                pad_token_id=processor.tokenizer.pad_token_id
            )

        # Extract tokens
        generated_ids = outputs[0].tolist()

        # Find BOI and EOI positions
        try:
            boi_pos = generated_ids.index(boi_token_id)
            eoi_pos = generated_ids.index(eoi_token_id, boi_pos + 1)

            # Extract image tokens (between BOI and EOI)
            image_tokens = generated_ids[boi_pos + 1:eoi_pos]

            if len(image_tokens) == 1024:
                all_image_tokens.append(image_tokens)
                print(f"  ✓ Extracted {len(image_tokens)} image tokens")
            else:
                print(f"  ✗ Invalid image token count: {len(image_tokens)}")
        except ValueError as e:
            print(f"  ✗ Failed to find BOI/EOI: {e}")

    print(f"\nSuccessfully generated {len(all_image_tokens)} samples")

    # Save results
    if output_file is None:
        output_file = os.path.join(repo_root, "attacks", f"target_{target_prompt.replace(' ', '_')[:30]}.json")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    result = {
        "target_prompt": target_prompt,
        "num_samples": len(all_image_tokens),
        "image_tokens_list": all_image_tokens,
        "config": {
            "boi_token_id": boi_token_id,
            "eoi_token_id": eoi_token_id,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Saved to: {output_file}")

    return all_image_tokens


def main():
    parser = argparse.ArgumentParser(description="Generate target image tokens for adversarial optimization")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--target_prompt", type=str, required=True, help="Target prompt (e.g., 'a photo of a dog')")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path")

    args = parser.parse_args()

    generate_target_image_tokens(
        model_path=args.model_path,
        target_prompt=args.target_prompt,
        num_samples=args.num_samples,
        device=args.device,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()
