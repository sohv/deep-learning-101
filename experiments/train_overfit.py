import torch, torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp import MLP
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
subset = Subset(train_data, range(100))  # tiny dataset
loader = DataLoader(subset, batch_size=16, shuffle=True)

model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

losses = []
for epoch in range(100):
    total_loss = 0
    for x, y in loader:
        out = model(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print(f"Epoch {epoch}, Loss: {total_loss:.2f}")

plt.plot(losses)
plt.title("Training Loss (Overfitting 100 Samples)")
plt.show()
writer.flush()
writer.close()