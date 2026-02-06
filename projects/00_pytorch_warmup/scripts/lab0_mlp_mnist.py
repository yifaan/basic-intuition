"""TODO: Lab 0 script to train an MLP on MNIST.

Goals:
- Parse CLI args (data dir, batch size, epochs, lr, overfit).
- Load MNIST with transforms.
- Build model, optimizer, and run training loop.
- Add an overfit mode for a tiny subset (e.g., 32 samples).
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from models.mlp import MLP
from utils.train import TrainConfig, train_one_epoch, eval_one_epoch

def main(args) -> None:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    if args.overfit:
        train_dataset = Subset(train_dataset, range(args.subset_size))
        test_dataset = Subset(test_dataset, range(args.subset_size))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = args.device
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = TrainConfig(lr=args.lr, epochs=args.epochs, log_every=1)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_one_epoch(model, optimizer, train_loader, device, config)
        acc = eval_one_epoch(model, test_loader, device)
        print(f"Validation Accuracy: {acc:.4f}")
