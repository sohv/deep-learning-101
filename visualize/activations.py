import matplotlib.pyplot as plt
import torch

def plot_layer_activations(model, x):
    hooks = []
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    for layer in model.model:
        if isinstance(layer, torch.nn.Linear):
            hooks.append(layer.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        model(x)

    for i, act in enumerate(activations):
        plt.figure()
        plt.hist(act.view(-1).numpy(), bins=100)
        plt.title(f"Activation Histogram Layer {i}")
    plt.show()

    for h in hooks:
        h.remove()