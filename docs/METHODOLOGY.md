# Attack Methodology: Technical Details

## Overview

This document explains the technical methodology of our cross-modal hijacking attack in detail.

---

## 1. Attack Pipeline

```
Phase 1: Soft Prompt Optimization
    ↓
Phase 2: Gradient-Guided Hardening
    ↓
Phase 3: Verification & Image Decoding
```

---

## 2. Phase 1: Soft Prompt Optimization

### 2.1 Objective

Find continuous embedding vectors that, when used as prefix, cause the model to generate target image tokens.

### 2.2 Input Structure (Teacher Forcing)

```
[instruction] [BOI] [soft_prefix] [EOI] [victim] [BOI] [target_image] [EOI]
     ↑          ↑        ↑          ↑       ↑        ↑        ↑
  System    Begin    32 vectors   End   "cat"   Begin   Dog tokens
  prompt    Image    (learnable)  Image         Image
```

### 2.3 Loss Function

Cross-entropy loss on predicting target image tokens:

```python
loss = CrossEntropy(
    model_logits[context_end : context_end + match_len],
    target_image_tokens[:match_len]
)
```

Where:
- `context_end` = len(instruction) + 1 + len(prefix) + 1 + len(victim) + 1
- `match_len` = 128 (first 128 image tokens)

### 2.4 Optimization

- **Optimizer**: AdamW
- **Learning rate**: 0.01 with cosine annealing
- **Epochs**: 150
- **Gradient clipping**: max_norm=1.0

### 2.5 Initialization

Start from target image tokens with small noise:
```python
soft_prefix = embedding(target_dog_tokens[:32]) + noise(0.02)
```

This provides a good starting point close to the solution.

---

## 3. Phase 2: Gradient-Guided Hardening

### 3.1 Challenge

Convert 32 continuous vectors (each 4096-dim) to discrete token IDs from vocabulary.

**Naive approach fails**: Simple nearest-neighbor loses 99.6% information.

### 3.2 Our Approach: HotFlip + Iterative Evaluation

For each position i:

```python
# 1. Compute FRESH gradient at current state
loss = compute_loss(current_embeddings)
loss.backward()
gradient = current_embeddings.grad[i]

# 2. HotFlip scoring
for token in image_vocabulary:
    score[token] = embedding[token] · (-gradient)

# 3. Select top-k candidates
candidates = top_k(scores, k=64)

# 4. Evaluate each candidate's actual loss
for candidate in candidates[:32]:
    probe = current_embeddings.clone()
    probe[i] = embedding[candidate]
    candidate_loss = compute_loss(probe)
    if candidate_loss < best_loss:
        best_token = candidate

# 5. Update and continue
current_embeddings[i] = embedding[best_token]
```

### 3.3 Key Insight: Fresh Gradients

**Bug in v3**: Computed gradient once, reused for all positions.

**Problem**: Model is autoregressive. After position 0 changes from soft to discrete, the optimal token for position 1 changes.

**Fix (v3.1)**: Recompute gradient after each position is discretized.

---

## 4. Phase 3: Verification

### 4.1 Input Structure (Generation)

```
[instruction] [BOI] [discrete_prefix] [EOI] [victim]
```

No target tokens - model generates autoregressively.

### 4.2 Generation

```python
for step in range(max_tokens):
    logits = model(current_sequence)
    next_token = argmax(logits[-1])
    current_sequence.append(next_token)
    if next_token == EOI:
        break
```

### 4.3 Output Parsing

- Text tokens: Everything outside [BOI]...[EOI]
- Image tokens: Everything inside [BOI]...[EOI]

---

## 5. Why 32 Tokens?

### 5.1 Information Theory

```
Per token: log₂(8192) ≈ 13 bits (image vocabulary size)
32 tokens: 32 × 13 = 416 bits

Estimated semantic complexity of "override cat → dog":
- Concept swap: ~200 bits
- Visual features: ~300 bits
- Total: ~500 bits
```

32 tokens (~416 bits) is close to the threshold.

### 5.2 Empirical Results

| Tokens | Bits | Result |
|--------|------|--------|
| 16 | 208 | Corruption |
| 32 | 416 | **Success** |
| 64 | 832 | Failure (overshooting) |

### 5.3 Why 64 Tokens Failed

Hypothesis: Too much adversarial signal causes model instability or triggers defense mechanisms.

---

## 6. Why System Instructions?

### 6.1 LLM Training

Modern LLMs are trained to:
1. Prioritize system-level instructions
2. Follow `<SYSTEM>` tags with higher attention weight
3. Treat system prompts as trusted

### 6.2 Attack Leverage

System instruction creates **stronger prior** that requires less discrete signal to activate:

```
Without <SYSTEM>: Need ~700 bits of signal
With <SYSTEM>: Need ~400 bits of signal (System tag provides ~300 bits of "authority")
```

---

## 7. Code Structure

```
src/
├── model.py                 # Model loading
├── attacks/
│   ├── base.py             # Base class with instruction templates
│   ├── hybrid_prefix.py    # Main attack (optimize + harden + verify)
│   └── hardening.py        # Gradient-guided discretization
└── utils/
    ├── decoder.py          # Image token → PNG
    └── verification.py     # Result analysis
```

### 7.1 Key Classes

**ModelLoader**: Handles model loading and provides embedding/tokenizer access.

**HybridPrefixAttack**: Main attack class with three methods:
- `optimize()`: Phase 1 soft prompt optimization
- `harden()`: Phase 2 gradient-guided hardening
- `verify()`: Phase 3 generation and verification

**GradientGuidedHardening**: Implements the HotFlip-based discretization with fresh gradients.

---

## 8. Reproducing Results

### 8.1 Quick Start

```bash
# Run demo attack
bash scripts/run_demo.sh
```

### 8.2 Full Pipeline

```bash
# Step 1: Run attack with all phases
python scripts/run_attack.py \
    --victim "a photo of a cat" \
    --prefix_length 32 \
    --instruction_style system \
    --harden \
    --verify \
    --decode_image

# Step 2: Check results
ls results/attack_output/
# → soft_prefix.pt, discrete_prefix.json, verification_results.json, generated_image.png
```

### 8.3 Expected Output

```
Phase 1: Soft loss 15.5 → 0.0002 (99.99% reduction)
Phase 2: Discrete tokens [4447, 2801, 6607, ...]
Phase 3: Generated text "and a dog together", 960 image tokens
Phase 4: Decoded image saved to generated_image.png
```

---

## 9. Troubleshooting

### 9.1 CUDA Out of Memory

Reduce `match_image_len` from 128 to 64.

### 9.2 Attack Generates Wrong Image

Check:
1. Is instruction_style set to "system"?
2. Is prefix_length at least 32?
3. Did optimization converge (loss < 0.01)?

### 9.3 Image Decoding Fails

Ensure image_tokens list is not empty and contains valid token IDs (4-8195).

---

## 10. Extending the Attack

### 10.1 Different Targets

Replace `target_dog.json` with tokens from any target image:

```python
# Generate target tokens
from src.model import ModelLoader
loader = ModelLoader("twgi-critique-anole-7b")
loader.load()

# Encode target image through VQ-GAN
target_tokens = encode_image("my_target.png")
```

### 10.2 Different Victims

Change `--victim` argument to any text prompt:

```bash
python scripts/run_attack.py --victim "a beautiful sunset"
```

### 10.3 Different Models

The attack methodology applies to any early-fusion multimodal model with:
- Unified text/image vocabulary
- VQ-based image representation
- Autoregressive generation

Adjust `image_token_ids` range and `boi_id`/`eoi_id` for different models.
