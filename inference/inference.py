import torch
import json
import argparse
import os
import sys

# Prefer the vendored Transformers in this repo if available
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_local_tf_src = os.path.join(_repo_root, "transformers", "src")
if os.path.isdir(_local_tf_src) and _local_tf_src not in sys.path:
    sys.path.insert(0, _local_tf_src)

from transformers import ChameleonConfig, ChameleonProcessor, ChameleonForConditionalGenerationWithCFG
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from PIL import Image


class StopAtSpecificTokenCriteria(StoppingCriteria):
    """Stop generation when a specific token is generated"""
    
    def __init__(self, stop_token_id, device):
        self.stop_token_id = stop_token_id
        self.device = device
    
    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last generated token is our stop token
        return (input_ids[0, -1] == self.stop_token_id).item()


class InterleavedGenerator:
    """Handles interleaved text-image generation with CFG switching"""
    
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.config = ChameleonConfig.from_pretrained(model_name)
        # Prefer FlashAttention 2 on GPUs; fall back to eager on CPU
        self.config._attn_implementation = "flash_attention_2" if self.device == "cuda" else "eager"
        
        self.processor = ChameleonProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"
        
        self.model = ChameleonForConditionalGenerationWithCFG.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=(torch.bfloat16 if self.device == "cuda" else torch.float32),
        ).to(self.device)
        
        # Get special tokens
        self.boi_token_id = self.config.boi_token_id
        self.eoi_token_id = self.config.eoi_token_id
        self.eos_token_id = self.config.eos_token_id
        self.pad_token_id = 1
        
        # Image vocabulary for filtering
        self.image_conditioned_allowed = set([i for i in range(4, 8196)]) | {
            self.config.bos_token_id,
            self.boi_token_id,
            self.eoi_token_id,
        }
        
        # Setup initial CFG (disabled for text)
        self.model.setup_cfg(
            guidance_scale_full=2.0,
            guidance_scale_image=1.2,
            guidance_scale_negative=0.0,
            guidance_scale_original_prompt=5.0,
            config=self.config,
            cfg_config="no"  # Start with no CFG for text
        )
        
        # Store original prompt tokens for CFG
        self.original_prompt_tokens = None
    
    def _prepare_cfg_batch(self, token_ids, cfg_type="normal"):
        """
        Prepare batch for CFG by creating multiple conditions
        
        Args:
            token_ids: Current token sequence
            cfg_type: Type of CFG ("normal", "obj", or "full")
            
        Returns:
            Tensor with multiple conditions for CFG
        """
        # Negative prompt for image generation
        negative_prompt = "text in the image, text, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
        negative_tokens = self.processor.tokenizer.encode(negative_prompt, add_special_tokens=False)
        
        batch_token_ids = []
        
        if cfg_type == "normal":
            # For normal CFG (3 conditions) - simpler version without original prompt
            # 1. Full condition
            batch_token_ids.append(token_ids)
            
            # 2. Unconditional (just BOI)
            batch_token_ids.append([self.boi_token_id])
            
            # 3. Image-conditioned tokens only
            image_only_tokens = [tok for tok in token_ids if tok in self.image_conditioned_allowed]
            if not image_only_tokens or image_only_tokens[-1] != self.boi_token_id:
                image_only_tokens.append(self.boi_token_id)
            batch_token_ids.append(image_only_tokens)
            
        elif cfg_type == "obj":
            # For object-focused generation (3 conditions)
            # 1. Full condition
            batch_token_ids.append(token_ids)
            
            # 2. Unconditional (just BOI)
            batch_token_ids.append([self.boi_token_id])
            
            # 3. Negative condition
            batch_token_ids.append(negative_tokens + [self.boi_token_id])
                
        elif cfg_type == "full":
            # For full CFG (5 conditions)
            # 1. Full condition
            batch_token_ids.append(token_ids)
            
            # 2. Image-conditioned tokens only
            image_only_tokens = [tok for tok in token_ids if tok in self.image_conditioned_allowed]
            if not image_only_tokens or image_only_tokens[-1] != self.boi_token_id:
                image_only_tokens.append(self.boi_token_id)
            batch_token_ids.append(image_only_tokens)
            
            # 3. Unconditional
            batch_token_ids.append([self.boi_token_id])
            
            # 4. Negative condition
            batch_token_ids.append(negative_tokens + [self.boi_token_id])
            
            # 5. Original prompt condition
            if self.original_prompt_tokens:
                orig_tokens = self.original_prompt_tokens.copy()
                if not orig_tokens or orig_tokens[-1] != self.boi_token_id:
                    orig_tokens.append(self.boi_token_id)
                batch_token_ids.append(orig_tokens)
            else:
                batch_token_ids.append([self.boi_token_id])
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in batch_token_ids)
        attention_masks = []
        
        for i, seq in enumerate(batch_token_ids):
            # Pad sequence
            padding_length = max_len - len(seq)
            if padding_length > 0:
                # Pad on the left (since padding_side="left")
                batch_token_ids[i] = [self.pad_token_id] * padding_length + seq
                attention_masks.append([0] * padding_length + [1] * len(seq))
            else:
                attention_masks.append([1] * len(seq))
        
        # Convert to tensors
        input_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self.device)
        
        return input_ids, attention_mask
    
    def generate_interleaved(
        self,
        prompt_tokens: list,
        original_prompt_tokens: list = None,
        max_length: int = 5000,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_images: int = 4,
        cfg_type: str = "normal",
        mode: str = "general"
    ):
        """
        Generate interleaved text and images by alternating between modes
        
        Args:
            prompt_tokens: Tokenized prompt
            original_prompt_tokens: Tokenized original prompt for CFG
            max_length: Maximum total generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_images: Maximum number of images to generate
            cfg_type: Type of CFG to use ("normal", "obj", or "full")
            mode: Generation mode for special handling
            
        Returns:
            Dictionary with final output and generation info
        """
        # Store original prompt tokens for CFG
        self.original_prompt_tokens = original_prompt_tokens if cfg_type == "full" else None
        
        # Keep track of all generated tokens
        all_tokens = prompt_tokens.copy()
        
        # Statistics
        num_images_generated = 0
        generation_segments = []
        
        # Continue generation until we hit EOS or max length
        while len(all_tokens) < max_length and num_images_generated < max_images:
            current_input_ids = torch.tensor([all_tokens], device=self.device)
            
            # Setup stopping criteria for BOI token
            stop_at_boi = StopAtSpecificTokenCriteria(self.boi_token_id, self.device)
            stop_at_eos = StopAtSpecificTokenCriteria(self.eos_token_id, self.device)
            
            # Disable CFG for text generation
            self.model.cfg_config = "no"
            
            if mode == "image_critique" and num_images_generated == 0:
                new_tokens = []
            else:
                text_output = self.model.generate(
                    input_ids=current_input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    stopping_criteria=StoppingCriteriaList([stop_at_boi, stop_at_eos]),
                    multimodal_generation_mode="interleaved-text-image",
                    pad_token_id=self.pad_token_id
                )
                # Extract newly generated tokens
                new_tokens = text_output[0][len(all_tokens):].tolist()
            
            all_tokens.extend(new_tokens)
            
            # Check if we stopped at EOS (end of generation)
            if all_tokens[-1] == self.eos_token_id:
                print("Reached end of sequence token.")
                break
            
            # Check if we stopped at BOI (need to generate image)
            if all_tokens[-1] == self.boi_token_id:
                print(f"Generating image {num_images_generated + 1}...")
                
                # Phase 2: Generate image with CFG
                # Determine the actual cfg_type to use
                actual_cfg_type = cfg_type
                
                # Special handling for object_thoughts mode
                if mode == "object_thoughts":
                    if num_images_generated < 2:
                        actual_cfg_type = "obj"
                    else:
                        actual_cfg_type = "full"
                
                # Enable CFG for image generation
                self.model.cfg_config = actual_cfg_type
                
                # Prepare CFG batch
                cfg_input_ids, cfg_attention_mask = self._prepare_cfg_batch(
                    all_tokens, cfg_type=actual_cfg_type
                )
                
                image_output = self.model.generate(
                    input_ids=cfg_input_ids,
                    attention_mask=cfg_attention_mask,
                    max_new_tokens=1026,
                    temperature=temperature,
                    do_sample=True,
                    multimodal_generation_mode="image-only",
                    pad_token_id=self.pad_token_id
                )
                
                # Extract only the first condition's output (the actual generation)
                new_image_tokens = image_output[0][len(cfg_input_ids[0]):].tolist()[:1025]
                all_tokens.extend(new_image_tokens)
                
                num_images_generated += 1
                
                # Disable CFG again for next text segment
                self.model.cfg_config = "no"
        
        return {
            "tokens": all_tokens,
            "num_images": num_images_generated,
            "total_length": len(all_tokens)
        }


def load_prompts_from_jsonl(file_path, mode, processor, model):
    """Load and tokenize prompts from JSONL file based on mode"""
    prompts = []
    prompt_tokens_list = []
    original_prompt_tokens_list = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            if mode == "image_critique" and "prompt" in data:
                prompts.append(data['prompt'])
                original_prompt_tokens_list.append(processor.tokenizer.encode(data['prompt'], add_special_tokens=False))
                prompt_tokens_list.append(processor.tokenizer.encode(f"Generate an image based on the given prompt. Then analyze whether the image matches the prompt, and generate a better image based on your analysis. The prompt is: {data['prompt']}", add_special_tokens=False) + [8710, 8197])
                
            elif mode == "object_thoughts" and "prompt" in data:
                prompts.append(data['prompt'])
                original_prompt_tokens_list.append(processor.tokenizer.encode(data['prompt'], add_special_tokens=False))
                prompt_tokens_list.append(processor.tokenizer.encode(f"Generate the objects in the prompt step by step, and then generate the complete image. The prompt is: {data['prompt']}", add_special_tokens=False) + [8710])
                
            elif mode == "general" and "prompt" in data:
                dev = next(model.parameters()).device
                use_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
                if 'images' in data:
                    images = [Image.open(img_path) for img_path in data['images']]
                    inputs = processor(
                        data['prompt'], images=images, padding=False, return_tensors="pt", return_for_text_completion=True
                    ).to(dev, dtype=use_dtype)
                    input_ids = inputs['input_ids']
                    if data['images'] is not None:
                        pixel_values = inputs["pixel_values"]
                        image_tokens = model.get_image_tokens(pixel_values)
                        special_image_mask = input_ids == 8711  # Image token ID
                        image_tokens = image_tokens.to(input_ids.device, input_ids.dtype)
                        input_ids = input_ids.masked_scatter(special_image_mask, image_tokens)
                else:
                    inputs = processor(
                        data['prompt'], padding=False, return_tensors="pt", return_for_text_completion=True
                    ).to(dev, dtype=use_dtype)
                    input_ids = inputs['input_ids']

                prompts.append(data['prompt'])
                original_prompt_tokens_list.append(input_ids[0].tolist() + [8710])
                prompt_tokens_list.append(input_ids[0].tolist() + [8710])
    
    return prompts, prompt_tokens_list, original_prompt_tokens_list


def save_results_to_jsonl(results, output_path):
    """Save results to JSONL file"""
    with open(output_path, 'w') as f:
        for result in results:
            # Convert to the expected format
            output = {
                "prompt": result.get("prompt", ""),
                "response": result["tokens"]
            }
            json.dump(output, f)
            f.write('\n')


def main():
    parser = argparse.ArgumentParser(description="Interleaved text-image generation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--mode", type=str, choices=["general", "image_critique", "object_thoughts"], default="general", help="Generation mode")
    parser.add_argument("--cfg_type", type=str, choices=["normal", "obj", "full"], default="normal", help="CFG type to use")
    parser.add_argument("--max_length", type=int, default=6144, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--output_file", type=str, default="output.jsonl", help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize generator
    print("Loading model...")
    generator = InterleavedGenerator(args.model_path)
    
    # Determine CFG type based on mode
    if args.mode == "image_critique":
        args.cfg_type = "full"  # Use full 5-condition CFG for critique mode
    elif args.mode == "object_thoughts":
        args.cfg_type = "obj"  # Start with obj, but will switch to full for third image
    # For general mode, use whatever cfg_type was specified (default: "normal")
    
    results = []
    
    # Process JSONL file
    prompts, prompt_tokens_list, original_prompt_tokens_list = load_prompts_from_jsonl(
        args.input_file, args.mode, generator.processor, generator.model.model
    )
    print(f"Loaded {len(original_prompt_tokens_list)} prompts from {args.input_file}")
    
    for i, (prompt, prompt_tokens, original_tokens) in enumerate(
        zip(prompts, prompt_tokens_list, original_prompt_tokens_list)
    ):
        print(f"\nProcessing prompt {i+1}/{len(original_prompt_tokens_list)}")
        result = generator.generate_interleaved(
            prompt_tokens=prompt_tokens,
            original_prompt_tokens=original_tokens,
            max_length=args.max_length,
            temperature=args.temperature,
            cfg_type=args.cfg_type,
            mode=args.mode
        )
        result["prompt"] = prompt
        results.append(result)
    
    # Save all results
    save_results_to_jsonl(results, args.output_file)
    
    # Print summary
    print(f"\nGeneration complete!")
    total_images = sum(r["num_images"] for r in results)
    print(f"Total prompts processed: {len(results)}")
    print(f"Total images generated: {total_images}")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
