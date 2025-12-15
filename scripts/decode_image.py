#!/usr/bin/env python3
"""
Decode image tokens from verification results to PNG image.

Usage:
    python scripts/decode_image.py results/attack_output/verification_results.json
    python scripts/decode_image.py results/attack_output/verification_results.json -o output.png
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from src.model import ModelLoader
from src.utils import ImageDecoder


def main():
    parser = argparse.ArgumentParser(description="Decode image tokens to PNG")
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to verification_results.json"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output image path (default: same directory as results)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="twgi-critique-anole-7b",
        help="Path to model weights"
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_file}")
    with open(args.results_file, 'r') as f:
        results = json.load(f)

    image_tokens = results.get('image_tokens', [])
    if not image_tokens:
        print("Error: No image tokens found in results")
        sys.exit(1)

    print(f"Found {len(image_tokens)} image tokens")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        results_dir = os.path.dirname(args.results_file)
        output_path = os.path.join(results_dir, "decoded_image.png")

    # Load model for decoder
    print("\nLoading model for decoding...")
    loader = ModelLoader(args.model_path)
    loader.load()

    # Decode
    decoder = ImageDecoder(loader)
    success = decoder.decode_and_save(image_tokens, output_path)

    if success:
        print(f"\nImage decoded and saved to: {output_path}")
    else:
        print("\nFailed to decode image")
        sys.exit(1)


if __name__ == "__main__":
    main()
