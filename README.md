# Normalising Flows on the Moons Dataset

M2 Deep Learning coursework, implementing affine coupling flows from scratch in PyTorch.

## Setup

Requires Python 3.13+.

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running

The entire pipeline lives in `coursework.ipynb`. Run all cells top-to-bottom:

```bash
jupyter notebook coursework.ipynb
```

This produces all required outputs: figures, checkpoints, training logs, and `results.json`.

## Project Structure

```
├── coursework.ipynb          # Main notebook, runs the full pipeline
├── writeup.md                # Written response (loaded into results.json)
├── best-practices.md         # Training guidelines from L07/L08
├── src/flows/
│   ├── model.py              # AffineCouplingLayer, Flow, SurgeryFlow
│   ├── data.py               # MoonsDataset, MoonsSplits, NormaliseStats
│   ├── train.py              # train_flow, evaluate_nll, checkpointing
│   ├── correctness.py        # Invertibility and log-det Jacobian checks
│   ├── sanity.py             # Initial loss and baseline sanity checks
│   ├── profile.py            # FLOP counting
│   ├── explore.py            # Summary statistics and KS tests
│   └── viz.py                # All plotting functions (Figure1c, 2a, 2c, 3b, etc.)
├── data/
│   ├── moons_train.csv       # 800 samples
│   ├── moons_val.csv         # 100 samples
│   └── moons_test.csv        # 100 samples
├── figs/                     # Generated figures (PDF)
├── checkpoints/              # Saved model weights
├── logs/                     # Training curve JSON
├── tests/                    # pytest tests
└── results.json              # Final results
```

## Model

An affine coupling flow with alternating masks:

- **Architecture**: `Linear(D→H) → ReLU → Linear(H→2D)` per coupling layer
- **Base density**: Standard normal N(0, I)
- **Surgery**: `SurgeryFlow` appends a shear transformation g_α([x1, x2]) = [x1 + αx2, x2] for post-hoc distribution manipulation

Constraints: CPU only, no external flow libraries, max 128 hidden units, max 8 layers, max 10k training steps.

## Outputs

| Artifact | Description |
|---|---|
| `results.json` | Correctness metrics, NLL scores, writeup |
| `figs/Figure1c.pdf` | Correctness checks (invertibility + log-det) |
| `figs/Figure2a.pdf` | Tiny-subset training curve |
| `figs/Figure2c.pdf` | Full training + validation curves |
| `figs/Figure3b.pdf` | Flow surgery samples (5 panels) |
| `checkpoints/flow_full.pt` | Model weights, config, seed |
| `logs/training_curves.json` | Per-step loss curves |
