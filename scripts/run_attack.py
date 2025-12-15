#!/usr/bin/env python3
"""
Main script to run cross-modal hijacking attack.

Usage:
    python scripts/run_attack.py --target_tokens configs/target_dog.json
    python scripts/run_attack.py --victim "a photo of a cat" --prefix_length 32
"""

import argparse
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from src.model import ModelLoader
from src.attacks import HybridPrefixAttack
from src.utils import ImageDecoder


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-modal hijacking attack on Anole-7B"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="twgi-critique-anole-7b",
        help="Path to model weights"
    )

    # Attack arguments
    parser.add_argument(
        "--target_tokens",
        type=str,
        default="configs/target_dog.json",
        help="Path to target image tokens JSON"
    )
    parser.add_argument(
        "--victim",
        type=str,
        default="a photo of a cat",
        help="Victim prompt to hijack"
    )
    parser.add_argument(
        "--prefix_length",
        type=int,
        default=32,
        help="Number of adversarial tokens (32 is optimal)"
    )
    parser.add_argument(
        "--instruction_style",
        type=str,
        default="system",
        choices=["system", "strong", "override", "short", "reasoning"],
        help="Instruction style (system is most effective)"
    )

    # Optimization arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=150,
        help="Number of optimization epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate"
    )
    parser.add_argument(
        "--match_image_len",
        type=int,
        default=128,
        help="Number of image tokens to match in loss"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/attack_output",
        help="Output directory"
    )

    # Action flags
    parser.add_argument(
        "--harden",
        action="store_true",
        help="Convert soft prefix to discrete tokens"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify attack by generating"
    )
    parser.add_argument(
        "--decode_image",
        action="store_true",
        help="Decode generated image tokens"
    )

    args = parser.parse_args()

    # Load target tokens
    target_path = os.path.join(PROJECT_ROOT, args.target_tokens)
    if not os.path.exists(target_path):
        # Try attacks directory as fallback
        target_path = os.path.join(PROJECT_ROOT, "attacks", "target_a_photo_of_a_dog.json")

    print(f"Loading target tokens from: {target_path}")
    with open(target_path, 'r') as f:
        target_data = json.load(f)
    target_tokens = target_data['image_tokens_list'][0]
    print(f"Loaded {len(target_tokens)} target image tokens")

    # Load model
    print("\n" + "="*70)
    print("Loading Model")
    print("="*70)
    loader = ModelLoader(args.model_path)
    loader.load()

    # Create attack
    attack = HybridPrefixAttack(
        loader,
        prefix_length=args.prefix_length,
        instruction_style=args.instruction_style,
    )

    # Run optimization
    print("\n" + "="*70)
    print("Phase 1: Soft Prompt Optimization")
    print("="*70)
    soft_prefix, opt_results = attack.optimize(
        victim_prompt=args.victim,
        target_image_tokens=target_tokens,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        match_image_len=args.match_image_len,
        output_dir=args.output_dir,
    )

    print(f"\nOptimization complete. Best loss: {opt_results['best_loss']:.6f}")

    # Harden to discrete tokens
    discrete_ids = None
    if args.harden:
        print("\n" + "="*70)
        print("Phase 2: Gradient-Guided Hardening")
        print("="*70)
        discrete_ids = attack.harden(
            soft_prefix,
            args.victim,
            target_tokens,
            match_image_len=args.match_image_len,
            output_dir=args.output_dir,
        )
        print(f"\nDiscrete prefix: {discrete_ids}")

    # Verify attack
    if args.verify and discrete_ids is not None:
        print("\n" + "="*70)
        print("Phase 3: Verification")
        print("="*70)
        verify_results = attack.verify(
            discrete_ids,
            args.victim,
            output_dir=args.output_dir,
        )

        # Decode image if requested
        if args.decode_image and verify_results['image_tokens']:
            print("\n" + "="*70)
            print("Phase 4: Image Decoding")
            print("="*70)
            decoder = ImageDecoder(loader)
            image_path = os.path.join(args.output_dir, "generated_image.png")
            success = decoder.decode_and_save(
                verify_results['image_tokens'],
                image_path
            )
            if success:
                print(f"Generated image saved to: {image_path}")

    print("\n" + "="*70)
    print("Attack Complete")
    print("="*70)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
