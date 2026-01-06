# MambaDerm

**Hybrid CNN-Mamba for Efficient Skin Lesion Classification**

A novel architecture combining ConvNeXt-style CNN stem for local features with Vision Mamba (VMamba) for O(n) global context modeling.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       MambaDerm (~31M params)                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐     ┌────────────────┐     ┌──────────────┐  │
│  │   CNN      │     │  Vision Mamba  │     │   Tabular    │  │
│  │   Stem     │────►│  (VMamba)      │────►│   Encoder    │  │
│  │ (ConvNeXt) │     │  4 layers      │     │              │  │
│  └────────────┘     └────────────────┘     └──────────────┘  │
│         │                   │                     │          │
│         └────────┬──────────┴──────────┬──────────┘          │
│                  ▼                     ▼                     │
│         ┌─────────────┐      ┌─────────────────┐             │
│         │ Local-Global│      │  Cross-Modal    │             │
│         │  Gating     │◄────►│  Attention      │             │
│         └─────────────┘      └─────────────────┘             │
│                          │                                   │
│                          ▼                                   │
│              ┌──────────────────────┐                        │
│              │     Classifier       │                        │
│              │     MLP Head         │                        │
│              └──────────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

- **Efficient Global Context**: O(n) complexity via State Space Models (Mamba)
- **Local-Global Fusion**: Adaptive gating between CNN and Mamba features
- **Multimodal**: Integrates 35 numerical + 6 categorical tabular features
- **Cross-Modal Attention**: Image features query tabular metadata
- **Production Ready**: AMP training, gradient checkpointing, pAUC@80% TPR

## Installation

```bash
cd MambaDerm
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for mamba-ssm)

```

### Training

```bash
# Train on fold 0
python train.py --fold 0 --epochs 30 --batch_size 32

# Quick test (sanity check)
python train.py --quick_test --epochs 1

# Full training with custom settings
python train.py \
    --fold 0 \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --d_model 192 \
    --n_mamba_layers 4 \
    --use_amp
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `isic-2024-challenge` | Path to ISIC data |
| `--fold` | 0 | CV fold (0-4) |
| `--epochs` | 30 | Number of epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--d_model` | 192 | Model dimension |
| `--n_mamba_layers` | 4 | Number of Mamba layers |
| `--use_amp` | True | Mixed precision training |
| `--quick_test` | False | Quick test mode |

## Project Structure

```
MambaDerm/
├── models/
│   ├── mambaderm.py          # Main model
│   ├── mamba_block.py        # Vision Mamba SSM
│   ├── cnn_stem.py           # ConvNeXt stem
│   ├── tabular_encoder.py    # Metadata encoder
│   └── local_global_gate.py  # Feature fusion
├── data/
│   ├── dataset.py            # ISIC dataset loader
│   ├── transforms.py         # Augmentation pipeline
│   └── sampler.py            # Balanced sampling
├── utils/
│   ├── metrics.py            # pAUC metric
│   ├── losses.py             # Focal/Asymmetric loss
│   └── scheduler.py          # LR schedulers
├── train.py                  # Training script
├── config.py                 # Configuration
└── requirements.txt          # Dependencies
```

## Model Variants

| Variant | d_model | Layers | Params | Memory |
|---------|---------|--------|--------|--------|
| MambaDerm | 192 | 4 | ~31M | ~8GB |
| MambaDerm-Lite | 128 | 2 | ~12M | ~4GB |

## License

MIT License
