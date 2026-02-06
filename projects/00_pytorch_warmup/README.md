# Phase 0 — PyTorch Warmup (Comprehensive)

## Objective
Rebuild fluency with Python + PyTorch fundamentals and training workflows across multiple model families.

## Scope
- MLP, CNN, Transformer
- From-scratch layers + pretrained usage
- Manual training loops, metrics, and debugging
- Tiny overfit checks for correctness

## Done when
- You can write a training loop from memory.
- You can overfit 32 samples with at least two model families.
- You can explain common failure modes (shape, device, dtype, optimization).
- You have one reusable training script that can swap models/datasets.

---

# Warmup Labs (checklist)

## Lab 0 — Python + PyTorch fundamentals
- [x] Implement a tiny MLP classifier from scratch.
- [x] Write a manual training loop (forward → loss → backward → step → zero_grad).
- [x] Overfit 32 samples and log loss/accuracy.
- **From scratch:** Linear + ReLU (no `nn.Linear`).
- **Done when:** loss approaches zero and gradients make sense.

## Lab 1 — Convolutional baseline
- [ ] Build a small CNN for MNIST/CIFAR.
- [ ] Track shapes at each layer and verify by hand.
- [ ] Overfit a tiny dataset subset.
- **From scratch:** naive `Conv2d` (or unfold + matmul) and compare outputs.
- **Done when:** stable loss decrease and correct shape reasoning.

## Lab 2 — Training utilities + hygiene
- [ ] Implement `train_one_epoch` and `eval_one_epoch`.
- [ ] Add checkpointing and simple logging.
- [ ] Create a clean config pattern (argparse or dataclass).
- **From scratch:** accuracy / top-k metrics.
- **Done when:** a single script can swap model/dataset cleanly.

## Lab 3 — Transformer mini
- [ ] Implement a tiny transformer for a toy task.
- [ ] Add positional encoding.
- [ ] Visualize or print attention weights for sanity checks.
- **From scratch:** scaled dot-product attention.
- **Done when:** toy task learns and attention shapes make sense.

## Lab 4 — Pretrained usage
- [ ] Load a pretrained backbone (e.g., ResNet).
- [ ] Replace the head and fine-tune on a small dataset.
- [ ] Freeze and unfreeze layers intentionally.
- **From scratch:** custom classification head.
- **Done when:** head training works, and fine-tuning improves results.

## Lab 5 — Detection flavors (lightweight)
- [ ] Implement a toy detector on a tiny dataset.
- [ ] Flavor A: anchor-based head (simplified SSD style).
- [ ] Flavor B: anchor-free head (center + size regression).
- **From scratch:** IoU, matching/assignment, NMS.
- **Done when:** can overfit a handful of images and interpret loss parts.

## Lab 6 — Scripting and tooling
- [ ] Custom `Dataset` and `DataLoader` with transforms.
- [ ] Seed control + deterministic flags.
- [ ] CLI entry point for each lab.
- **From scratch:** small data caching or preprocessing routine.
- **Done when:** you can run any lab with a single command.

---

# Suggested Order
1. Lab 0 → Lab 1 → Lab 2
2. Lab 3 → Lab 4
3. Lab 5 → Lab 6

# Notes
Log all observations in `LEARNING.md` with date-stamped entries.
