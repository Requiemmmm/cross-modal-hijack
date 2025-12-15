#!/usr/bin/env python3
"""
Experiment 1: Pixel-level Verification
======================================
将32个VISUAL token IDs解码为adversarial image patch

目标：
1. 证明VISUAL tokens = adversarial image patches（可以作为图像上传）
2. 验证这是真正的black-box attack（不需要token ID API）
3. 测试re-encode的tokens是否仍然有效

流程：
Visual Token IDs → VQ-GAN Decode → Adversarial Patch (PNG)
                                           ↓
                    VQ-GAN Encode → Token IDs' → Test Attack
"""

import sys
sys.path.insert(0, "/root/autodl-tmp/thinking-with-generated-images/transformers/src")
sys.path.insert(0, "/root/autodl-tmp/thinking-with-generated-images")

import torch
import json
import os
from transformers import ChameleonConfig, ChameleonProcessor, ChameleonForConditionalGenerationWithCFG
from transformers.image_transforms import to_pil_image
from PIL import Image
import numpy as np

def decode_visual_tokens_to_patch(model, processor, visual_token_ids):
    """
    将32个visual tokens解码为adversarial patch

    注意：32 tokens = 32x8x8像素块（使用VQ-GAN的8x8 patch size）
          但我们需要将其排列成图像
    """
    print("="*60)
    print("PHASE 1: DECODE VISUAL TOKENS TO IMAGE PATCH")
    print("="*60)
    print()

    print(f"Visual token IDs: {visual_token_ids[:10]}... ({len(visual_token_ids)} total)")

    # 检查token范围
    assert all(4 <= tid <= 8195 for tid in visual_token_ids), "Some tokens are not VISUAL tokens!"
    print(f"✓ All {len(visual_token_ids)} tokens are VISUAL tokens (4-8195)")
    print()

    # Chameleon的图像token是1024个（32x32 grid，每个8x8像素）
    # 但我们只有32个tokens，所以需要padding或者创建一个小图像

    # 策略：创建32x32 token grid，但只用前32个位置，其余填充为black (token 4)
    BLACK_TOKEN = 4  # 假设token 4是黑色或neutral

    # 构造1024 token序列
    full_tokens = [BLACK_TOKEN] * 1024
    full_tokens[:32] = visual_token_ids

    print(f"Creating 32x32 token grid (1024 tokens total)")
    print(f"  - First 32 positions: adversarial tokens")
    print(f"  - Remaining 992 positions: black/neutral (token {BLACK_TOKEN})")
    print()

    # 解码
    token_tensor = torch.tensor([full_tokens], device="cuda", dtype=torch.long)

    print("Decoding via VQ-GAN...")
    with torch.no_grad():
        pixel_values = model.model.decode_image_tokens(token_tensor)
        images = processor.postprocess_pixel_values(pixel_values)

    patch_image = to_pil_image(images[0].detach().cpu())

    print(f"✓ Decoded to image: {patch_image.size}")
    print()

    return patch_image, full_tokens

def encode_patch_to_tokens(model, processor, patch_image):
    """
    将adversarial patch重新编码为tokens
    """
    print("="*60)
    print("PHASE 2: RE-ENCODE IMAGE PATCH TO TOKENS")
    print("="*60)
    print()

    print(f"Input image size: {patch_image.size}")

    # 预处理图像
    pixel_values = processor.image_processor(patch_image, return_tensors="pt")['pixel_values']
    pixel_values = pixel_values.to(device="cuda", dtype=torch.bfloat16)

    print(f"Pixel values shape: {pixel_values.shape}")

    # 编码
    print("Encoding via VQ-GAN...")
    with torch.no_grad():
        encoded_output = model.model.vqmodel.encode(pixel_values)
        # encode返回tuple: (quantized, indices)
        if isinstance(encoded_output, tuple):
            image_tokens = encoded_output[1]  # indices
        else:
            image_tokens = encoded_output

    print(f"  Image tokens shape: {image_tokens.shape}")
    print(f"  Image tokens dtype: {image_tokens.dtype}")

    # Convert to list
    if image_tokens.dim() == 0:
        reencoded_ids = [image_tokens.item()]
    elif image_tokens.dim() == 1:
        reencoded_ids = image_tokens.cpu().tolist()
    else:
        # Flatten if needed
        reencoded_ids = image_tokens.flatten().cpu().tolist()

    print(f"✓ Re-encoded to {len(reencoded_ids)} tokens")
    print(f"  Token range: [{min(reencoded_ids)}, {max(reencoded_ids)}]")
    print()

    return reencoded_ids

def test_reencoded_attack(model, processor, reencoded_ids, instruction, victim_prompt):
    """
    测试re-encoded tokens是否仍然有效
    """
    print("="*60)
    print("PHASE 3: TEST RE-ENCODED ATTACK")
    print("="*60)
    print()

    BOI = 8197
    EOI = 8196

    # 只使用前32个re-encoded tokens（对应我们的adversarial patch）
    adversarial_tokens = reencoded_ids[:32]

    print(f"Using first 32 re-encoded tokens as adversarial prefix")
    print(f"  Tokens: {adversarial_tokens[:10]}...")
    print()

    # Tokenize instruction and victim
    instruction_ids = processor.tokenizer(instruction, add_special_tokens=False)['input_ids']
    victim_ids = processor.tokenizer(victim_prompt, add_special_tokens=False)['input_ids']

    # Construct input: instruction + adversarial_tokens + victim + [BOI]
    input_ids = instruction_ids + adversarial_tokens + victim_ids + [BOI]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")

    print(f"Input structure:")
    print(f"  - Instruction: {len(instruction_ids)} tokens")
    print(f"  - Adversarial patch: {len(adversarial_tokens)} tokens")
    print(f"  - Victim: {len(victim_ids)} tokens")
    print(f"  - Total: {len(input_ids)} tokens")
    print()

    print("Generating with re-encoded adversarial patch...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_tensor,
            max_new_tokens=1050,
            do_sample=False,
            bos_token_id=BOI,
            eos_token_id=EOI,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    new_tokens = generated_ids[0, len(input_ids):]

    # Split image and text
    try:
        eoi_idx = (new_tokens == EOI).nonzero(as_tuple=True)[0][0].item()
        image_tokens = new_tokens[:eoi_idx].cpu().tolist()
        text_tokens = new_tokens[eoi_idx+1:]
    except:
        image_tokens = new_tokens[:1024].cpu().tolist()
        text_tokens = []

    generated_text = processor.tokenizer.decode(text_tokens, skip_special_tokens=True) if len(text_tokens) > 0 else ""

    print(f"✓ Generated {len(image_tokens)} image tokens")
    print(f"✓ Generated text: \"{generated_text}\"")
    print()

    # Evaluation
    print("="*60)
    print("ATTACK EVALUATION")
    print("="*60)
    print()

    success = False
    if "dog" in generated_text.lower() and "cat" not in generated_text.lower():
        print("🎯 SUCCESS! Re-encoded adversarial patch still works!")
        print("   This proves VISUAL tokens = uploadable adversarial patches")
        success = True
    elif "cat" in generated_text.lower():
        print("❌ FAILED: Attack signal lost during re-encoding")
    else:
        print("⚠️  PARTIAL: Need to check the image manually")

    print()

    return {
        'reencoded_tokens': adversarial_tokens,
        'generated_text': generated_text,
        'image_tokens': image_tokens,
        'success': success
    }

def compare_tokens(original_ids, reencoded_ids):
    """
    比较原始tokens和re-encoded tokens
    """
    print("="*60)
    print("TOKEN COMPARISON")
    print("="*60)
    print()

    # 只比较前32个
    orig_32 = original_ids[:32]
    reen_32 = reencoded_ids[:32]

    matches = sum(1 for o, r in zip(orig_32, reen_32) if o == r)
    match_rate = matches / 32

    print(f"Original tokens (first 32): {orig_32[:10]}...")
    print(f"Re-encoded tokens (first 32): {reen_32[:10]}...")
    print()
    print(f"Exact matches: {matches}/32 ({match_rate*100:.1f}%)")
    print()

    if match_rate > 0.9:
        print("✓ High fidelity: VQ-GAN encode-decode is nearly lossless")
    elif match_rate > 0.5:
        print("⚠️ Moderate fidelity: Some quantization loss")
    else:
        print("❌ Low fidelity: Significant information loss")

    print()

    return match_rate

def main():
    print("="*60)
    print("EXPERIMENT 1: PIXEL-LEVEL VERIFICATION")
    print("="*60)
    print()
    print("Research Question:")
    print("  Can VISUAL tokens be decoded to an adversarial image patch")
    print("  and uploaded as a TRUE black-box attack?")
    print()

    # Load model
    model_path = "twgi-critique-anole-7b"
    print(f"Loading model: {model_path}")

    config = ChameleonConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"
    processor = ChameleonProcessor.from_pretrained(model_path)
    model = ChameleonForConditionalGenerationWithCFG.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    model.setup_cfg(cfg_config="no")

    print("✓ Model loaded")
    print()

    # Load successful 32-token attack
    prefix_file = "outputs/reproducibility/prefix_32_run2/hybrid_discrete_prefix.json"
    print(f"Loading successful attack prefix: {prefix_file}")

    with open(prefix_file, 'r') as f:
        prefix_data = json.load(f)

    visual_token_ids = prefix_data['visual_discrete_ids']
    instruction = prefix_data['instruction_text']

    print(f"✓ Loaded {len(visual_token_ids)} visual tokens")
    print(f"  Instruction: {instruction}")
    print()

    # Phase 1: Decode to patch
    patch_image, full_token_sequence = decode_visual_tokens_to_patch(model, processor, visual_token_ids)

    # Save patch
    output_dir = "outputs/visual_patch_experiment"
    os.makedirs(output_dir, exist_ok=True)

    patch_path = os.path.join(output_dir, "adversarial_patch.png")
    patch_image.save(patch_path)
    print(f"✓ Adversarial patch saved: {patch_path}")
    print()

    # Phase 2: Re-encode patch
    reencoded_ids = encode_patch_to_tokens(model, processor, patch_image)

    # Compare tokens
    match_rate = compare_tokens(full_token_sequence, reencoded_ids)

    # Phase 3: Test attack with re-encoded tokens
    victim_prompt = "a photo of a cat"
    result = test_reencoded_attack(model, processor, reencoded_ids, instruction, victim_prompt)

    # Save re-encoded results
    result_data = {
        'original_visual_tokens': visual_token_ids,
        'reencoded_tokens_32': result['reencoded_tokens'],
        'full_reencoded_tokens_1024': reencoded_ids,
        'token_match_rate': match_rate,
        'generated_text': result['generated_text'],
        'image_tokens': result['image_tokens'],
        'success': result['success']
    }

    result_file = os.path.join(output_dir, "reencoded_attack_results.json")
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"✓ Results saved: {result_file}")
    print()

    # Decode generated image
    if len(result['image_tokens']) == 1024:
        print("Decoding generated image...")
        token_tensor = torch.tensor([result['image_tokens']], device="cuda", dtype=torch.long)
        with torch.no_grad():
            pixel_values = model.model.decode_image_tokens(token_tensor)
            images = processor.postprocess_pixel_values(pixel_values)

        generated_image = to_pil_image(images[0].detach().cpu())
        generated_path = os.path.join(output_dir, "reencoded_generated_image.png")
        generated_image.save(generated_path)
        print(f"✓ Generated image saved: {generated_path}")
        print()

    # Final summary
    print("="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print()

    if result['success']:
        print("🎉 BREAKTHROUGH: Visual tokens ARE adversarial image patches!")
        print()
        print("Key findings:")
        print(f"  1. Visual tokens decoded to uploadable PNG image")
        print(f"  2. Re-encoding preserved {match_rate*100:.1f}% of tokens")
        print(f"  3. Re-encoded patch still triggers dog generation")
        print()
        print("Implication: TRUE black-box attack via image upload!")
    else:
        print("📊 Results:")
        print(f"  - Token match rate: {match_rate*100:.1f}%")
        print(f"  - Generated text: \"{result['generated_text']}\"")
        print(f"  - Attack success: {'Yes' if result['success'] else 'No'}")
        print()
        print("The adversarial patch exists, but may need refinement.")

    print()

if __name__ == "__main__":
    main()
