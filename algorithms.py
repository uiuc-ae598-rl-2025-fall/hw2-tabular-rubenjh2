# Acknowledgement: Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). The MIT Press.
# If an input is defined in an earlier function docstring, it is not re-defined in later functions.
import numpy as np


# shared epsilon greedy policy function
def eps_greedy_a(Q, s, epsilon=0.2):
    '''
    Epsilon-greedy policy for action selection

    New Inputs:
    Q: (num_s, num_a) array of action-value function
    s: current state
    epsilon: probability of choosing random action

    Returns:
    a: action selected for that state
    '''
    num_a = Q.shape[1]

    if np.random.rand() < epsilon:
        # if random float (0 to 1) < epsilon, choose random action as best
        a = np.random.randint(num_a)
    else:
        q_s = Q[s]
        max_q = q_s.max()
        # get indices of tied Q values
        best_actions = np.flatnonzero(q_s == max_q)
        # action for the state corresponds to highest Q value (w/ tiebreaks)
        a = np.random.choice(best_actions)

    return a


def run_episode(env, Q, epsilon=0.2, max_steps=100):
    '''
    Run an episode for MC Control

    New inputs:
    env: OpenAI Gym environment
    max_steps: max steps per episode

    Returns:
    path: list of (state, action, reward) tuples
    '''
    # reset env to new episode, get initial state
    s, _ = env.reset()
    steps = 0
    path = []

    for n in range(max_steps):
        a = eps_greedy_a(Q, s, epsilon)
        # run a step, get next state, reward, terminated, truncated, _ (dont need auxiliary info)
        s_next, r, terminated, truncated, _ = env.step(a)
        steps += 1

        path.append((s, a, r))

        if terminated or truncated:
            break

        s = s_next

    return path, steps

def MC_control(env, Q_init=None, count_init=None, num_episodes=5000, epsilon=0.1, gamma=0.95, max_steps=1000):
    '''
    First-visit MC Control with epsilon-greedy policy.

    New Inputs:
    Q_init: (num_s, num_a) array of initial action-value function for warm start
    count_init: (num_s, num_a) array of initial visit counts for warm start
    num_episodes: number of episodes to run
    gamma: discount factor

    Returns:
    Q: (num_s, num_a) array of learned action-value function
    MC_steps: total environment steps taken during training (for evaluation return plotting)
    count: (num_s, num_a) array of state-action visit counts
    '''
    num_s = env.observation_space.n
    num_a = env.action_space.n
    MC_steps = 0

    # warm start for checkpoint evaluation returns
    Q = Q_init.copy() if Q_init is not None else np.zeros((num_s, num_a))
    count = count_init.copy() if count_init is not None else np.zeros((num_s, num_a))

    for _ in range(num_episodes):
        G = 0.0
        visited = set()

        path, ep_steps = run_episode(env, Q, epsilon=epsilon, max_steps=max_steps)

        MC_steps += ep_steps

        # returns depend on future rewards so iterating in reverse allows one pass to update G
        for s, a, r in reversed(path):
            G = r + gamma * G

            # first visit only
            if (s, a) in visited:
                continue

            visited.add((s, a))
            count[s, a] += 1

            # incremental mean
            Q[s, a] += (G - Q[s, a]) / count[s, a]
    
    return Q, MC_steps, count



def SARSA(env, Q_init=None, num_episodes=1000, epsilon=0.2, gamma=0.95, alpha=0.1, max_steps=1000):
    '''
    On-policy TD(0) SARSA

    New Inputs:
    alpha: learning rate

    Returns:
    Q: (num_s, num_a) array of learned action-value function
    sarsa_steps: total environment steps taken during training (for evaluation return plotting)
    '''
    sarsa_steps = 0

    num_s = env.observation_space.n
    num_a = env.action_space.n

    Q = Q_init.copy() if Q_init is not None else np.zeros((num_s, num_a))

    for _ in range(num_episodes):
        # get first state and action and reset
        s, _ = env.reset()
        a = eps_greedy_a(Q, s, epsilon)

        # TD(0), so update after every step
        for _ in range(max_steps):

            s_next, r, terminated, truncated, _ = env.step(a)
            sarsa_steps += 1

            if not (terminated or truncated):
                a_next = eps_greedy_a(Q, s=s_next, epsilon=epsilon)

                Q[s, a] += alpha * (r + gamma*Q[s_next, a_next] - Q[s, a])

            else:
                # if terminal, no next state or action so AV func simplifies
                Q[s, a] += alpha * (r - Q[s, a])
                break

            s, a = s_next, a_next

    return Q, sarsa_steps



def Q_learning(env, num_episodes=1000, Q_init=None, epsilon=0.2, gamma=0.95, alpha=0.1, max_steps=1000):
    '''
    Off-policy TD(0) Q-learning

    New Inputs: None

    Returns:
    Q: (num_s, num_a) array of learned action-value function
    Q_steps: total environment steps taken during training (for evaluation return plotting)
    '''
    Q_steps = 0

    num_s = env.observation_space.n
    num_a = env.action_space.n
    # Apply "warm start" for checkpoint evaluation returns
    Q = Q_init.copy() if Q_init is not None else np.zeros((num_s, num_a))

    for _ in range(num_episodes):
        s, _ = env.reset()
        for _ in range(max_steps):
            a = eps_greedy_a(Q, s, epsilon=epsilon)

            s_next, r, terminated, truncated, _ = env.step(a)
            Q_steps += 1

            if not (terminated or truncated):
                Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
                s = s_next
            else:
                Q[s, a] += alpha * (r - Q[s, a])
                break

    return Q, Q_steps