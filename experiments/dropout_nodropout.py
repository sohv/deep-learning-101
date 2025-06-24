import torch, torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from models.mlp import MLP

train = torchvision.datasets.MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test = torchvision.datasets.MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=64)

models = {"dropout": MLP(use_dropout=True), "no_dropout": MLP(use_dropout=False)}

for label, model in models.items():
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    print(f"Training: {label}")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss:.2f}")