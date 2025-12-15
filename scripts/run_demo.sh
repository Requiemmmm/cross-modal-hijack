#!/bin/bash
# Demo script for running cross-modal hijacking attack
#
# This script demonstrates the successful attack configuration:
# - 32 visual tokens (phase transition threshold)
# - System-level instruction
# - Prefix positioning
#
# Usage: bash scripts/run_demo.sh

set -e

echo "=========================================="
echo "Cross-Modal Hijacking Attack Demo"
echo "=========================================="
echo ""
echo "Attack Configuration:"
echo "  - Victim prompt: 'a photo of a cat'"
echo "  - Target: Generate dog image instead"
echo "  - Prefix length: 32 tokens"
echo "  - Instruction: <SYSTEM> tags"
echo ""

cd "$(dirname "$0")/.."

python scripts/run_attack.py \
    --model_path twgi-critique-anole-7b \
    --target_tokens configs/target_dog.json \
    --victim "a photo of a cat" \
    --prefix_length 32 \
    --instruction_style system \
    --num_epochs 150 \
    --learning_rate 0.01 \
    --output_dir results/demo_output \
    --harden \
    --verify \
    --decode_image

echo ""
echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo "Results saved to: results/demo_output/"
echo "  - soft_prefix.pt: Optimized soft embeddings"
echo "  - discrete_prefix.json: Hardened discrete tokens"
echo "  - verification_results.json: Generation results"
echo "  - generated_image.png: Decoded image"
