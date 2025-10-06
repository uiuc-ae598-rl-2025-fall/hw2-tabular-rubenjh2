# If an input is defined in an earlier function docstring, it is not re-defined in later functions.
import numpy as np
from algorithms import MC_control, SARSA, Q_learning

def pi_from_Q(env, Q):
    '''
    Get the greedy policy pi from Q.

    New Inputs:
    env: OpenAI Gym environment
    Q: (num_s, num_a) array of action-value function

    Returns:
    pi: (num_s,) policy array where pi[s] is the best action for state s
    '''
    num_s = env.observation_space.n

    # one action for every state
    pi = np.zeros((num_s))

    for s in range(num_s):
        q_s = Q[s]
        max_q = q_s.max()
        best_actions = np.flatnonzero(q_s == max_q)
        pi[s] = np.random.choice(best_actions)
    
    return pi

def evaluation_return(env, Q, gamma=0.95, max_steps=1000, num_episodes=1000):
    '''
    Evaluate the average return of the policy derived from Q by running num_episodes. 

    New Inputs:
    gamma: discount factor
    max_steps: max steps per evaluation episode
    num_episodes: number of evaluation episodes
    '''
    pi = pi_from_Q(env, Q=Q)
    G_list = []

    for _ in range(num_episodes):
        s, _ = env.reset()
        G = 0.0
        for i in range(max_steps):
            a = int(pi[s])
            s, r, terminated, truncated, _ = env.step(a)

            G += gamma**(i) * r

            if terminated or truncated:
                break
        
        G_list.append(G)
    G_episode = np.mean(G_list)

    return G_episode


def train_all_checkpoints(env, final_num_episodes=10000, alpha=0.05, gamma=0.95):
    '''
    Train all algorithms and return evaluation returns at checkpoints.

    New Inputs:
    final_num_episodes: total number of episodes to train each algorithm
    alpha: learning rate for TD methods

    Returns:
    dict of results for each algorithm, each containing:
    "steps": (num_checkpoints,) array of environment steps at each checkpoint
    "returns": (num_checkpoints,) array of evaluation returns at each checkpoint
    "Q": (num_s, num_a) array of learned action-value function
    '''
    # Evaluation after checkpoints many episodes
    checkpoints = np.linspace(0, final_num_episodes, 100, dtype=int)

    eval_gamma, eval_max_steps, num_episodes = gamma, 500, 500

    # MC Control
    mc_steps_list, mc_returns_list = [], []
    Q_mc, total_mc_steps, total_mc_count, prev = None, 0, None, 0

    for n_eps in checkpoints:
        add = n_eps - prev 
        prev = n_eps

        Q_mc, used, MC_count = MC_control(env, Q_init=Q_mc, count_init=total_mc_count, num_episodes=add, epsilon=0.2, gamma=gamma, max_steps=1000)
        total_mc_steps += used
        total_mc_count = MC_count

        G = evaluation_return(env, Q_mc, gamma=eval_gamma, max_steps=eval_max_steps, num_episodes=num_episodes)

        mc_steps_list.append(total_mc_steps); mc_returns_list.append(G)

    # SARSA
    sa_steps_list, sa_returns_list = [], []
    Q_sa, total_sa_steps, prev = None, 0, 0

    for n_eps in checkpoints:
        add = n_eps - prev 
        prev = n_eps

        Q_sa, used = SARSA(env, Q_init=Q_sa, num_episodes=add, epsilon=0.2, gamma=gamma, alpha=alpha, max_steps=1000)
        total_sa_steps += used

        G = evaluation_return(env, Q_sa, gamma=eval_gamma, max_steps=eval_max_steps, num_episodes=num_episodes)

        sa_steps_list.append(total_sa_steps); sa_returns_list.append(G)

    # Q learning
    q_steps_list, q_returns_list = [], []
    Q_q, total_q_steps, prev = None, 0, 0

    for n_eps in checkpoints:
        add = n_eps - prev
        prev = n_eps

        Q_q, used = Q_learning(env, Q_init=Q_q, num_episodes=add, epsilon=0.2, gamma=gamma, alpha=alpha, max_steps=1000)
        total_q_steps += used

        G = evaluation_return(env, Q_q, gamma=eval_gamma, max_steps=eval_max_steps, num_episodes=num_episodes)

        q_steps_list.append(total_q_steps); q_returns_list.append(G)

    return {
        "MC":    {"steps": np.array(mc_steps_list), "returns": np.array(mc_returns_list), "Q": Q_mc},
        "SARSA": {"steps": np.array(sa_steps_list), "returns": np.array(sa_returns_list), "Q": Q_sa},
        "Q":     {"steps": np.array(q_steps_list),  "returns": np.array(q_returns_list),  "Q": Q_q}}