# Phase 1 — Object Detection (Perception)

## Objective
Understand modern detection pipelines and loss design.

## Scope
- Pretrained backbone + custom detection head
- Small, manageable dataset
- Focus on correctness and stability, not SOTA numbers

## Key concepts to learn
- Bounding box parameterization
- Classification vs localization losses
- IoU, matching/assignment strategies
- Non-Maximum Suppression (NMS)
- Basic evaluation metrics (precision, recall, mAP intuition)

## Done when
- Detector can overfit a tiny dataset
- Produces reasonable bounding boxes on validation images
- Losses are stable and interpretable
- You understand why it fails when it fails

## Action items
- Pick a small dataset subset (e.g., VOC subset) and define a clean split.
- Build a detector with pretrained backbone + lightweight head.
- Implement loss breakdown and log each component.
- Add a tiny-dataset overfit check with 2–5 images.
- Visualize predictions and failure cases.
