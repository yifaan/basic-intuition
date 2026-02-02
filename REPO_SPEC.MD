# ML Learning Lab — Repo Intent

## Purpose
This repository is a personal ML learning lab.

Goals:
- Rebuild strong Python + PyTorch intuition through small, complete projects
- Focus on robotics-relevant problems (perception, decision making, data efficiency)
- Stay close to modern ML practice without chasing scale
- Prefer understanding training dynamics over building huge models

Primary compute:
- Local Ubuntu desktop with NVIDIA GTX 1060
- Cloud GPU used only when scale or VRAM becomes limiting

---

## Learning philosophy
- Small projects that finish
- Each project should have:
  - a clear objective
  - a minimal implementation
  - a concrete “done when” condition
- Write training loops, losses, metrics manually
- Use pretrained weights where helpful
- Add abstractions only when repetition appears

---

## High-level roadmap (projects, not code)

### Phase 0 — PyTorch refresh
Goal:
- Become fluent again with PyTorch training loops and debugging

Examples:
- Simple image classification
- Overfitting tiny datasets
- Clean training loop with checkpointing and logging

---

### Phase 1 — Object detection (robotics perception)
Goal:
- Understand detection losses, assignment, and evaluation

Scope:
- Pretrained backbone + custom detection head
- Small datasets (VOC / subset)
- Emphasis on loss design and stability

---

### Phase 2 — Diffusion fundamentals
Goal:
- Understand diffusion objectives, schedules, and sampling

Scope:
- DDPM-style model on small datasets
- Focus on training dynamics, not image quality

---

### Phase 3 — Reinforcement learning basics
Goal:
- Learn RL training dynamics and failure modes

Scope:
- Classic control tasks (CartPole, etc.)
- Implement core algorithms manually
- Focus on stability and reward design

---

### Phase 4 — Active learning
Goal:
- Learn how data selection affects model efficiency

Scope:
- Pool-based active learning on detection
- Compare uncertainty-based selection vs random

---

### Phase 5 — Gaussian splatting (exploratory)
Goal:
- Understand modern 3D scene representations relevant to robotics

Scope:
- Conceptual understanding + small-scale experiments
- No requirement for full reproduction

---

## Repo structure (minimal, flexible)

ml-learning-lab/
  README.md
  ROADMAP.md
  LEARNING.md
  projects/
    00_pytorch_warmup/
    10_detection/
    20_diffusion/
    30_rl/
    40_active_learning/
    50_gaussian_splatting/

Each project directory should be self-contained and free to evolve independently.

---

## Tracking progress
- Keep ROADMAP.md high-level and stable
- Keep LEARNING.md as a living log:
  - what worked
  - what failed
  - what surprised me
- Use git history and commits as the main record of progress

---

## What this repo is NOT
- A production ML framework
- A benchmark-chasing codebase
- A perfectly abstracted library

It is a learning laboratory.

END
