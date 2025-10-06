from env import build_env
from train import train_all_checkpoints
from plot import plot_eval, plot_policy, plot_value


def main():
    gamma = 0.95

    # Build slippery and non-slippery FrozenLake env
    env_slip   = build_env(success_rate=1.0/3.0, is_slippery=True)
    env_noslip = build_env(success_rate=1.0/3.0, is_slippery=False)

    for env, tag in [(env_slip,   "(slippery=True)"), (env_noslip, "(slippery=False)")]:
        alpha = 0.025
        # Train all algos with checkpoints for evaluation return plotting
        results = train_all_checkpoints(env, final_num_episodes=20000, alpha=alpha, gamma=gamma)

        # Plot eval return vs time steps
        series = {"MC":    
                  (results["MC"]["steps"],    results["MC"]["returns"]),
                  "SARSA": 
                  (results["SARSA"]["steps"], results["SARSA"]["returns"]),
                  "Q":     
                  (results["Q"]["steps"],     results["Q"]["returns"])}

        plot_eval(series, show=False, title=f"[{tag}] alpha={alpha} Evaluation Return — All Methods")

        # Policy plots
        plot_policy(env, results["MC"]["Q"], title=f"Policy (MC Control) [{tag}]", show=False)
        plot_policy(env, results["SARSA"]["Q"], title=f"Policy (SARSA) [{tag}]", show=False)
        plot_policy(env, results["Q"]["Q"], title=f"Policy (Q-learning) [{tag}]", show=False)

        # Heatmaps
        plot_value(env, results["MC"]["Q"], title=f"V(s) (MC Control) [{tag}]", show=False)
        plot_value(env, results["SARSA"]["Q"], title=f"V(s) (SARSA) [{tag}]", show=False)
        plot_value(env, results["Q"]["Q"], title=f"V(s) (Q-learning) [{tag}]", show=False)

    print("\nDone with all training and plotting.")


if __name__ == "__main__":
    main()
