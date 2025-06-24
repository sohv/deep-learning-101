import matplotlib.pyplot as plt

def plot_gradients(model):
    norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms.append(param.grad.norm().item())

    plt.bar(range(len(norms)), norms)
    plt.title("Gradient Norms per Layer")
    plt.xlabel("Layer Index")
    plt.ylabel("Gradient Norm")
    plt.show()