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
from dataclasses import dataclass
import torch
from torch import nn

# TODO: implement TrainConfig
# 1) @dataclass
# 2) fields: lr: float, epochs: int, log_every: int
@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 10
    log_every: int = 100

# TODO: implement accuracy
# 1) def accuracy(logits, targets) -> float
# 2) preds = logits.argmax(dim=1)
# 3) return mean correctness as float
def accuracy(logits, targets) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).float().mean().item()
    return correct

# TODO: implement train_one_epoch
# 1) set model.train()
# 2) define loss_fn = nn.CrossEntropyLoss()
# 3) for each batch: move to device, forward, loss, backward, step
# 4) log loss/acc every log_every steps
def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer,
                    dataloader: torch.utils.data.DataLoader,
                    device: torch.device, config: TrainConfig) -> None:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for step, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        if (step + 1) % config.log_every == 0:
            acc = accuracy(logits, targets)
            print(f"Step {step+1}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

# TODO: implement eval_one_epoch
# 1) set model.eval(), torch.no_grad()
# 2) compute average loss and accuracy
# 3) return average accuracy
def eval_one_epoch(model: nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   device: torch.device) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy(logits, targets) * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    print(f"Eval Loss: {avg_loss:.4f}, Eval Acc: {avg_acc:.4f}")
    return avg_acc
