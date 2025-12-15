# Cross-Modal Hijacking Attack on Early-Fusion Multimodal Models

Security research demonstrating adversarial attacks on Chameleon/Anole-7B multimodal models.

## Key Achievement

Successfully demonstrated **discrete cross-modal hijacking**: using 32 adversarial tokens to override text instructions and generate different visual content.

```
Input:  "a photo of a cat"
Output: Dog image (instead of cat)
```

## Quick Start

```bash
# 1. Install dependencies
bash install.sh

# 2. Download model (if not present)
huggingface-cli download GAIR/twgi-critique-anole-7b --local-dir twgi-critique-anole-7b

# 3. Run attack demo
bash scripts/run_demo.sh
```

## Attack Configuration

The successful attack uses three critical factors:

| Factor | Value | Rationale |
|--------|-------|-----------|
| **Instruction Style** | `<SYSTEM>` tags | Priority attention mechanism |
| **Token Count** | 32 | Phase transition threshold (~416 bits) |
| **Position** | Prefix | Early instructions establish stronger priors |

## Results

| Metric | Value |
|--------|-------|
| Soft optimization loss | 0.0002 |
| Initial discrete loss | 0.0723 |
| Generated image | Clear dog (Corgi) |
| Reproducibility | 100% (2/2 runs) |

## Project Structure

```
.
├── src/                          # Core attack code
│   ├── model.py                  # Model loading utilities
│   ├── attacks/
│   │   ├── hybrid_prefix.py      # Main attack (successful method)
│   │   ├── hardening.py          # Soft→Discrete conversion
│   │   └── base.py               # Base attack class
│   └── utils/
│       ├── decoder.py            # Image token decoder
│       └── verification.py       # Result verification
├── scripts/
│   ├── run_attack.py             # Main attack script
│   ├── run_demo.sh               # Demo script
│   └── decode_image.py           # Image decoding utility
├── configs/
│   └── target_dog.json           # Target image tokens
├── docs/
│   ├── EXPERIMENT_REPORT.md      # Complete experiment report
│   └── METHODOLOGY.md            # Technical methodology
├── results/                      # Attack results
├── outputs/                      # Experiment outputs (legacy)
└── attacks/                      # Legacy attack implementations
```

## Documentation

- **[docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)** - Complete experiment report with all findings
- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** - Technical methodology and implementation details

## Usage

### Basic Attack

```bash
python scripts/run_attack.py \
    --victim "a photo of a cat" \
    --prefix_length 32 \
    --instruction_style system \
    --harden \
    --verify \
    --decode_image
```

### Programmatic Usage

```python
from src.model import ModelLoader
from src.attacks import HybridPrefixAttack
from src.utils import ImageDecoder

# Load model
loader = ModelLoader("twgi-critique-anole-7b")
loader.load()

# Create attack
attack = HybridPrefixAttack(loader, prefix_length=32, instruction_style="system")

# Run attack
soft_prefix, _ = attack.optimize(
    victim_prompt="a photo of a cat",
    target_image_tokens=target_tokens,
)

# Harden to discrete tokens
discrete_ids = attack.harden(soft_prefix, "a photo of a cat", target_tokens)

# Verify
results = attack.verify(discrete_ids, "a photo of a cat")

# Decode image
decoder = ImageDecoder(loader)
decoder.decode_and_save(results['image_tokens'], "output.png")
```

## Key Findings

### 1. Phase Transition at 32 Tokens

| Tokens | Information | Result |
|--------|-------------|--------|
| 16 | 208 bits | Corruption (threshold edge) |
| **32** | **416 bits** | **Success** |
| 64 | 832 bits | Failure (overshooting) |

### 2. Modality Caste System

- **Visual tokens** have semantic override privilege
- **Pure text tokens** cannot achieve cross-modal hijacking
- System instructions amplify visual token effects

### 3. Multiplicative Effect

Three factors must **combine** for success:
- System instruction alone → Corruption
- More tokens alone → Likely failure
- **System + 32 tokens + Prefix** → Success!

## Security Implications

### Threat Model

Adversarial visual tokens can be:
1. Provided as token IDs (gray-box)
2. Decoded to PNG and uploaded as image (true black-box)

### Defense Recommendations

1. Limit visual prefix length
2. Validate/sanitize system instructions
3. Monitor unusual image token patterns
4. Implement semantic consistency checks

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (16GB+ VRAM recommended)
- Model weights: `twgi-critique-anole-7b`

## Citation

If you use this research, please cite:

```bibtex
@misc{crossmodal-hijacking-2024,
  title={Cross-Modal Hijacking Attack on Early-Fusion Multimodal Models},
  year={2024},
  note={Security research on Chameleon/Anole-7B}
}
```

## Acknowledgments

This research builds upon:
- [Thinking with Generated Images](https://arxiv.org/abs/2505.22525) - Base model and framework
- [LARGO](https://arxiv.org/abs/2401.XXXXX) - Inspiration for soft-to-discrete conversion
- [HotFlip](https://arxiv.org/abs/1712.06751) - Gradient-based token selection

## License

Research code follows the same license as the base model ([Chameleon License](https://ai.meta.com/resources/models-and-libraries/chameleon-license)).

---

**Disclaimer**: This research is for educational and defensive security purposes only. The techniques demonstrated should only be used for authorized security testing and improving model robustness.
