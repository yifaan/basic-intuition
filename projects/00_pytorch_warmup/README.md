# Phase 0 — PyTorch Warmup

## Objective
Rebuild fluency with Python + PyTorch fundamentals and training workflows.

## Scope
- Simple supervised learning (e.g., MNIST / CIFAR)
- Explicit training loops
- GPU usage (CUDA)
- Debugging and sanity checks

## Key concepts to learn
- Tensors, modules, optimizers, losses
- Device placement (CPU vs GPU)
- Train vs eval mode
- Overfitting tiny datasets as a correctness check

## Done when
- Can write a training loop from memory
- Loss decreases reliably
- Can overfit a very small dataset (e.g., 32 samples)
- Comfortable debugging shape / device / dtype issues

## Action items
- Implement a minimal classifier with a manual training loop.
- Add a tiny-dataset overfit test (e.g., 32 samples) as a correctness check.
- Log loss/accuracy to verify training dynamics.
- Add a short debugging checklist in this folder once you hit common issues.
