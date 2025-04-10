from Optimizers import Optimizer
from Functions import Function
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Learning rates to test
learning_rates = [0.01, 0.0025, 0.001]
init_param = [-1, 0]
function = Function.rosenbrock_value
gradient = Function.rosenbrock_gradient

# Define grid for contour plot
x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(x, y)
Z = function(X, Y)  # Compute function values on the grid

optimizers = {
    "Adam": lambda opt: opt.adam,
    "Gradient Descent": lambda opt: opt.gd,
    "Stochastic Gradient Descent": lambda opt: opt.sgd,
    "SGD with Momentum": lambda opt: opt.sgdm,
    "Nesterov Accelerated Gradient": lambda opt: opt.nesterov,
    "Adagrad": lambda opt: opt.adagrad
}

colors = plt.cm.tab10(np.linspace(0, 1, len(optimizers)))  # Assign unique colors

# Create subplots for each learning rate (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loop through each learning rate and plot in the corresponding subplot
for i, lr in enumerate(learning_rates):
    ax = axes[i]  # Select the current subplot (axes[i])
    
    # Run optimization with current learning rate
    optimizer = Optimizer(function=gradient, learning_rate=lr)

    norm = mcolors.LogNorm(vmin=Z.min(), vmax=Z.max())
    ax.contourf(X, Y, Z, levels=100, cmap="coolwarm", alpha=0.7, norm=norm)  

    for (name, opt_func), color in zip(optimizers.items(), colors):
        opt_function = opt_func(optimizer)  # Get the optimizer method
        params, trajectory = opt_function(init_params=init_param)
        print(f"Optimized Parameters ({name}) with lr={lr}:", params)

        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.6, linewidth=3, label=name)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, edgecolors='black', s=50)

    # Denote the theoretical minimum
    ax.scatter(3, 0.5, s=200, facecolors='none', edgecolors='orange', alpha = 0.5, linewidths=2, marker='*', label = "Saddle point")

    # Add legend to the last plot
    if i == 2:
            ax.legend(loc="lower right")

    # Set plot limits
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title(f"Optimization Trajectories (lr={lr})")
    ax.grid(False)

# Adjust layout to avoid overlap
plt.suptitle("Beale function")
plt.tight_layout()
plt.show()