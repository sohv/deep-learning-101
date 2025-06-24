import torch, torchvision
from models.mlp import MLP
from visualize.gradient_norms import plot_gradients

data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

x, y = next(iter(loader))
out = model(x)
loss = loss_fn(out, y)
optimizer.zero_grad()
loss.backward()

plot_gradients(model)