"""TODO: Lab 0 script to train an MLP on MNIST.

Goals:
- Parse CLI args (data dir, batch size, epochs, lr, overfit).
- Load MNIST with transforms.
- Build model, optimizer, and run training loop.
- Add an overfit mode for a tiny subset (e.g., 32 samples).
"""

# TODO: add imports
# - argparse, pathlib
# - torch
# - torchvision.datasets, torchvision.transforms
# - DataLoader, Subset
# - your MLP + training utilities

# TODO: parse_args
# 1) data-dir, batch-size, epochs, lr, overfit flag, subset size, device
# 2) set device default to cuda if available

# TODO: main
# 1) build transforms (ToTensor + Normalize)
# 2) load MNIST train/test
# 3) if overfit: Subset both to first N samples
# 4) create loaders
# 5) init model + optimizer
# 6) loop over epochs: train_one_epoch + eval_one_epoch
