# For image critique mode
python inference.py \
    --model_path ../twgi-critique-anole-7b \
    --input_file prompts_image_generation.jsonl \
    --mode image_critique \
    --output_file critique_output.jsonl


# For object thoughts mode  
python inference.py \
    --model_path ../twgi-subgoal-anole-7b \
    --input_file prompts_image_generation.jsonl \
    --mode object_thoughts \
    --output_file object_output.jsonl

# For general prompt testing
python inference.py \
    --model_path ../anole-7b-hf-2025 \
    --input_file prompts_general.jsonl \
    --mode general \
    --cfg_type normal \
    --output_file general_output.jsonl

# Phase 1: simple cat image generation (object thoughts)
python inference.py \
    --model_path ../twgi-critique-anole-7b \
    --input_file prompts_phase1.jsonl \
    --mode object_thoughts \
    --cfg_type obj \
    --output_file phase1_object_output.jsonl
