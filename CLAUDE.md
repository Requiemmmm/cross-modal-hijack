# Cross-Modal Hijacking Attack Research

## Project Overview

Security research on cross-modal hijacking attacks against early-fusion multimodal models (Chameleon/Anole-7B).

**Goal**: Find adversarial discrete tokens that override text instructions to generate different visual content.

**Status**: ✅ **SUCCESS** - 32-token attack achieves cat→dog hijacking.

## Quick Reference

### Run Attack
```bash
# Full attack pipeline
python scripts/run_attack.py \
    --victim "a photo of a cat" \
    --prefix_length 32 \
    --instruction_style system \
    --harden --verify --decode_image

# Or use demo script
bash scripts/run_demo.sh
```

### Key Configuration (Successful Attack)
- **Tokens**: 32 (phase transition threshold)
- **Instruction**: `<SYSTEM>` tags
- **Position**: Prefix (before victim prompt)

## Project Structure

```
.
├── src/                      # Core attack code
│   ├── model.py              # Model loading
│   ├── attacks/
│   │   ├── hybrid_prefix.py  # Main attack (successful)
│   │   ├── hardening.py      # Soft→Discrete conversion
│   │   └── base.py           # Base class
│   └── utils/
│       ├── decoder.py        # Image decoder
│       └── verification.py   # Result verification
├── scripts/
│   ├── run_attack.py         # Main script
│   ├── run_demo.sh           # Demo
│   └── decode_image.py       # Image decoding
├── configs/
│   └── target_dog.json       # Target tokens
├── docs/
│   ├── EXPERIMENT_REPORT.md  # Full report
│   └── METHODOLOGY.md        # Technical details
├── results/                  # Attack results
│   ├── successful_attack/    # 32-token success
│   └── reproducibility/      # Reproducibility test
├── outputs/                  # Legacy experiment outputs
├── attacks/                  # Legacy attack implementations
├── inference/                # Image generation utilities
├── transformers/             # Modified HuggingFace (required)
└── twgi-critique-anole-7b/   # Model weights
```

## Model Info

- **Model**: Chameleon/Anole-7B
- **BOI token**: 8197
- **EOI token**: 8196
- **Image tokens**: 1024 per image (32×32 grid)
- **Image vocabulary**: tokens 4-8195

## Key Results

| Configuration | Result |
|---------------|--------|
| 16 tokens + System | Corruption (threshold edge) |
| **32 tokens + System** | **✅ Success (generates dog)** |
| 64 tokens + System | Failure (overshooting) |
| 32 tokens + no System | Failure |
| Pure text tokens | Failure (modality lock) |

## Code Entry Points

### Main Attack
```python
from src.model import ModelLoader
from src.attacks import HybridPrefixAttack

loader = ModelLoader("twgi-critique-anole-7b")
loader.load()

attack = HybridPrefixAttack(loader, prefix_length=32, instruction_style="system")
soft_prefix, _ = attack.optimize("a photo of a cat", target_tokens)
discrete_ids = attack.harden(soft_prefix, "a photo of a cat", target_tokens)
results = attack.verify(discrete_ids, "a photo of a cat")
```

### Legacy Attack Code
Located in `attacks/` directory for reference:
- `hybrid_reasoning_prefix.py` - Original successful implementation
- `language_bridge_attack.py` - Language bridge (white-box only)
- `largo_v2_improved.py` - LARGO implementation
- `visual_prefix_attack.py` - Visual prefix experiments

## Documentation

- **[docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)** - Complete experiment report
- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** - Technical methodology
- **[README.md](README.md)** - Project overview and quick start
