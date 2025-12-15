python detokenization.py \
    --model_path ../twgi-subgoal-anole-7b \
    --jsonl_path ./object_output.jsonl \
    --output_dir ./decoded_output_obj/ \
    --device cuda

python detokenization.py \
    --model_path ../twgi-critique-anole-7b \
    --jsonl_path ./critique_output.jsonl \
    --output_dir ./decoded_output_critique/ \
    --device cuda

python detokenization.py \
    --model_path ../anole-7b-hf-2025 \
    --jsonl_path ./general_output.jsonl \
    --output_dir ./decoded_output_general/ \
    --device cuda

# Phase 1: decode cat image run
python detokenization.py \
    --model_path ../twgi-critique-anole-7b \
    --jsonl_path ./phase1_object_output.jsonl \
    --output_dir ./decoded_output_phase1/ \
    --device cuda
