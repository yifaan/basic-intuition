# ML Learning Lab — Repo Spec (Ethan)

## Goal
Build a personal ML learning repo that:
1) refreshes Python + PyTorch fluency via small, complete projects,
2) stays robotics-relevant (perception → decision → data selection),
3) keeps me close to state of the art without chasing scale,
4) runs primarily on my local Ubuntu desktop w/ NVIDIA GTX 1060 (6GB typical), with optional cloud “scale-up” runs only when necessary.

## Core principles
- Small projects that finish (each has a deliverable + “done when” checks).
- CPU-first debugging; GPU used for short, intentional training bursts.
- “Low-level” means: I implement the training loop, losses, metrics, data pipeline. I can use pretrained backbones/weights.
- Reproducible experiments: config files, seeds, logging, checkpoints.

---

## Repo structure (create these folders/files)

ml-learning-lab/
  README.md
  ROADMAP.md
  LEARNING.md
  ENVIRONMENT.md
  LICENSE (MIT)

  src/
    __init__.py
    utils/
      __init__.py
      seed.py
      logging.py
      checkpoint.py
      meters.py
      config.py
      device.py
    data/
      __init__.py
      common.py
    models/
      __init__.py
    trainers/
      __init__.py
      supervised_trainer.py
      rl_trainer.py
    eval/
      __init__.py

  projects/
    00_pytorch_warmup/
      README.md
      train_mnist.py
      configs/
        base.yaml
    10_detection_tiny/
      README.md
      train_detector.py
      configs/
        base.yaml
      src_detection/
        dataset.py
        model.py
        losses.py
        metrics.py
        nms.py
        visualize.py
    20_diffusion_mini/
      README.md
      train_ddpm.py
      sample_ddpm.py
      configs/
        base.yaml
      src_diffusion/
        schedule.py
        unet.py
        ema.py
        losses.py
        sampling.py
    30_rl_basics/
      README.md
      train_cartpole_dqn.py
      train_cartpole_pg.py
      configs/
        dqn.yaml
        pg.yaml
      src_rl/
        envs.py
        replay.py
        networks.py
        losses.py
        utils.py
    40_active_learning_detection/
      README.md
      active_loop.py
      configs/
        base.yaml
      src_active/
        selectors.py
        uncertainty.py
        loop.py
    50_gaussian_splatting_tiny/
      README.md
      notes.md
      (placeholder scaffolding only; no heavy implementation required initially)

  scripts/
    setup_venv.sh
    lint.sh
    format.sh
    run_tensorboard.sh

  tests/
    test_seed.py
    test_checkpoint.py
    test_train_step_smoke.py

  requirements.txt
  pyproject.toml (optional; include basic tool config)
  Makefile

---

## Implementation details (what to put in the shared utilities)

### src/utils/seed.py
- set_seed(seed): sets python/random/numpy/torch seeds and torch.backends flags
- ensures reproducibility where possible

### src/utils/logging.py
- standard python logging configured with timestamps
- optional TensorBoard support via torch.utils.tensorboard.SummaryWriter
- a function get_writer(log_dir) returning SummaryWriter

### src/utils/checkpoint.py
- save_checkpoint(path, model, optimizer, scaler(optional), step, config)
- load_checkpoint(path, model, optimizer(optional), scaler(optional))

### src/utils/config.py
- load YAML config into dict
- merge CLI overrides in the form key=value (simple)

### src/utils/device.py
- choose device (cuda if available else cpu)
- print GPU name if cuda

### src/trainers/supervised_trainer.py
- a generic supervised training loop that supports:
  - AMP (torch.cuda.amp.autocast + GradScaler)
  - grad clipping
  - checkpoint saving every N steps
  - logging loss + lr
- Expose a train() function that each project can reuse.

### src/trainers/rl_trainer.py
- minimal RL loop utilities (rollout, update steps, logging)

### tests/test_train_step_smoke.py
- simple test that runs a tiny model forward/backward and ensures loss decreases for a few steps (on CPU).

---

## Dependencies
requirements.txt should include at least:
- torch
- torchvision
- numpy
- pyyaml
- tqdm
- tensorboard
- opencv-python (optional; for detection visualization)
- matplotlib

Optional dev tools:
- ruff
- black
- pytest

---

## Makefile targets
- make setup         (create venv + install requirements)
- make test          (pytest)
- make format        (black)
- make lint          (ruff)
- make tb            (tensorboard --logdir runs)

---

## ENVIRONMENT.md (include concise setup)
### Local machine (Ubuntu + GTX 1060)
- Install NVIDIA driver + CUDA-compatible PyTorch build (keep it simple; rely on pip torch build)
- Verify:
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

### Optional cloud escalation
- Only use cloud if VRAM/time limits block progress.
- Keep code identical; only change config (batch size, num workers, etc).

---

## ROADMAP.md (the learning plan)

### Phase 0 — PyTorch refresh (Projects/00_pytorch_warmup)
**Objective:** regain fluency with PyTorch training loops and debugging.
Tasks:
- implement a clean train loop with AMP + checkpoint + TensorBoard
- overfit 32 samples on MNIST (prove correctness)
Done when:
- training loss reliably decreases on tiny subset
- checkpoints resume cleanly

### Phase 1 — Tiny detection (Projects/10_detection_tiny)
**Objective:** learn detection losses/assignment/metrics with pretrained backbone.
Scope:
- Use a pretrained backbone (ResNet18/34) from torchvision
- Start frozen backbone; train a small head
- Add unfreeze stage later
Implement:
- IoU computation
- NMS
- box loss (L1 + GIoU or DIoU)
- classification loss (CE; optionally focal)
- basic evaluation (precision/recall @ IoU threshold)
Dataset:
- start with a small, easy dataset (Pascal VOC or a tiny subset)
Done when:
- can overfit a tiny set (8–32 images)
- produces reasonable boxes on val sample images
- training is stable (no NaNs, no exploding loss)

### Phase 2 — Mini diffusion (Projects/20_diffusion_mini)
**Objective:** understand diffusion objective, schedules, and sampling.
Scope:
- DDPM on CIFAR-10 or small custom dataset
Implement:
- forward noising schedule
- reverse sampling loop
- EMA of weights
Done when:
- samples qualitatively improve over time
- loss curve stable and reproducible
Optional stretch:
- classifier-free guidance basics

### Phase 3 — RL basics (Projects/30_rl_basics)
**Objective:** understand RL training dynamics, credit assignment, instability.
Scope:
- CartPole with:
  - DQN (experience replay)
  - REINFORCE / policy gradient baseline
Implement:
- replay buffer
- target network (for DQN)
- advantage normalization / baseline
Done when:
- can solve CartPole with DQN and one policy gradient variant
- can explain failure modes (exploration, reward scaling, variance)

### Phase 4 — Active learning loop on detection (Projects/40_active_learning_detection)
**Objective:** learn data selection strategies and measure label efficiency.
Scope:
- pool-based active learning
Loop:
1) train detector on small labeled set
2) infer on unlabeled pool
3) select top-K by uncertainty
4) (simulate labeling or manual)
5) retrain
Selectors:
- entropy / margin
- disagreement (if using dropout or ensembling-lite)
Done when:
- active selection beats random on at least one metric for same label budget
- selection logic is clearly documented

### Phase 5 — Gaussian splatting (Projects/50_gaussian_splatting_tiny)
**Objective:** learn the idea, data flow, and core optimization loop conceptually.
Scope:
- Keep this as notes + small-scale reproduction attempt only after prior phases
Done when:
- can run a tiny scene or reproduce a minimal pipeline OR can explain algorithm stages with diagrams/notes

---

## LEARNING.md (how progress is tracked)
- Keep an “Active Projects” section with 1–2 items max.
- Keep a “Learned / Notes” section with brief bullets (what surprised me, key debugging lessons).
- Keep links to:
  - important commits
  - checkpoint artifacts (if stored)
  - experiment configs

Template:

# Active Projects
- [ ] P10 Detection Tiny — next task: implement GIoU loss

# Notes (keep short)
- YYYY-MM-DD: Overfit harness caught a label encoding bug.

# Links
- Detection baseline commit: <hash>
- TensorBoard run: runs/...

---

## README.md (top-level quickstart)
Include:
- what this repo is
- how to setup venv
- how to run each project
- how to view TensorBoard

Example commands:
- make setup
- make test
- python projects/00_pytorch_warmup/train_mnist.py --config projects/00_pytorch_warmup/configs/base.yaml
- tensorboard --logdir runs

---

## Scripts
scripts/setup_venv.sh:
- create .venv
- pip install -r requirements.txt

scripts/run_tensorboard.sh:
- tensorboard --logdir runs --port 6006 --bind_all (optional)

---

## Coding style and conventions
- Prefer clarity over cleverness
- Keep functions small
- Document “done when” criteria in each project README
- Always include an overfit mode in supervised projects

---

## Initial GitHub Issues to create (optional but recommended)
- P00: PyTorch Warmup (owner: Ethan)
- P10: Tiny Detector v1
- P20: DDPM Mini
- P30: RL CartPole DQN + PG
- P40: Active Learning Loop for Detector
- P50: Gaussian Splatting Notes + Tiny Attempt

Each issue should include:
- goal
- checklist
- done when
- notes section

END

