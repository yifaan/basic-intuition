"""TODO: Training utilities for Phase 0 labs.

Goals:
- Implement TrainConfig (epochs, lr, log_every).
- Implement accuracy metric.
- Implement train_one_epoch and eval_one_epoch.
"""

# TODO: add imports
# - from dataclasses import dataclass
# - import torch
# - from torch import nn

# TODO: implement TrainConfig
# 1) @dataclass
# 2) fields: lr: float, epochs: int, log_every: int

# TODO: implement accuracy
# 1) def accuracy(logits, targets) -> float
# 2) preds = logits.argmax(dim=1)
# 3) return mean correctness as float

# TODO: implement train_one_epoch
# 1) set model.train()
# 2) define loss_fn = nn.CrossEntropyLoss()
# 3) for each batch: move to device, forward, loss, backward, step
# 4) log loss/acc every log_every steps

# TODO: implement eval_one_epoch
# 1) set model.eval(), torch.no_grad()
# 2) compute average loss and accuracy
# 3) return average accuracy
