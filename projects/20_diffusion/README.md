# Phase 2 — Diffusion Models

## Objective
Understand diffusion as a generative modeling paradigm.

## Scope
- Small diffusion model (DDPM-style)
- Small image dataset
- Emphasis on training dynamics and sampling logic

## Key concepts to learn
- Forward noising process
- Reverse denoising / sampling
- Noise schedules
- EMA of model weights
- Why diffusion training is stable

## Done when
- Loss curve behaves as expected
- Generated samples improve qualitatively over training
- You can explain each step of the diffusion pipeline

## Action items
- Implement a minimal DDPM with a small UNet-style backbone.
- Add a simple noise schedule and verify forward noising.
- Log training loss and sample grids at fixed intervals.
- Add a short explanation of the sampling loop in this README.
