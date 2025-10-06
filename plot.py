# If an input is defined in an earlier function docstring, it is not re-defined in later functions.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from train import pi_from_Q


# Plotting
def plot_eval(series, show=True, title="Evaluation Return — All Methods"):
    '''
    Plot evaluation returns for multiple methods.

    New Inputs:
    series: dict of {name: (num_steps, returns)}
    show: if True, display the plot
    title: title of the plot
    '''
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, (steps, rets) in series.items():
        ax.plot(np.asarray(steps), np.asarray(rets), linewidth=1.8, label=name)

    ax.set_title(title)
    ax.set_xlabel("Environment time steps")
    ax.set_ylabel("Evaluation return")
    ax.grid(True, alpha=0.35)
    ax.legend()
    plt.tight_layout()

    plt.savefig(f"{title}.png")

    if show:
        plt.show()

    plt.close(fig)


# Plot the policy that corresponds to each trained agent
def plot_policy(env, Q, title="Policy", show=True):
    '''
    Make plot of policy derived from Q.

    New Inputs:
    env: OpenAI Gym environment
    Q: (num_s, num_a) array of action-value function

    '''
    action_mapping = {0:'←', 1:'↓', 2:'→', 3:'↑'}
    pi = pi_from_Q(env, Q)

    desc = env.unwrapped.desc
    mapping = {b'S': 0, b'F': 1, b'H': 2, b'G': 3}
    numeric = np.vectorize(mapping.get)(desc)
    H, W = numeric.shape

    colors = ['lightgreen', 'lightblue', 'darkblue', 'red']
    labels = ['Start (S)', 'Frozen (F)', 'Hole (H)', 'Goal (G)']
    cmap = ListedColormap(colors)

    # Terminal cells, H or G
    terminal = np.isin(numeric, [2, 3])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(numeric, origin='upper', cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.grid(True, color='k', alpha=0.15, linewidth=1)
    ax.set_title(title)

    # Arrows
    for r in range(H):
        for c in range(W):
            if not terminal[r, c]:
                s = r * W + c
                a = int(pi[s])
                ax.text(c, r, action_mapping[a], ha='center', va='center',
                        color='k', fontsize=20)

    handles = [Patch(facecolor=colors[i], edgecolor='k', label=labels[i]) for i in range(4)]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.12),
              ncol=4, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{title}.png")

    if show:
        plt.show()

    plt.close(fig)

def plot_value(env, Q, title="State Value", show=True):
    '''
    Make heatmap of greedy state values V(s) = max_a Q(s,a).

    New Inputs: None
    '''
    V = Q.max(axis=1)

    desc = env.unwrapped.desc
    H, W = desc.shape
    V_grid = V.reshape(H, W)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(V_grid, origin="upper")

    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.grid(True, color="k", alpha=0.15, linewidth=1)
    ax.set_title(title)

    for r in range(H):
        for c in range(W):
            cell = desc[r, c]
            if cell not in (b'H', b'G'):
                ax.text(c, r, f"{V_grid[r, c]:.2f}",
                        ha="center", va="center", color="k", fontsize=11)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("V(s)")

    plt.tight_layout()
    plt.savefig(f"{title}.png")
    if show:
        plt.show()
    plt.close(fig)