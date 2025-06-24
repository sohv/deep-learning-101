import torch, torch.nn as nn
import torchvision
from models.mlp import MLP
from visualize.activations import plot_layer_activations

data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

model = MLP()

x, _ = next(iter(loader))
plot_layer_activations(model, x)