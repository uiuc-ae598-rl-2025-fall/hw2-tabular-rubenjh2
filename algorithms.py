# Acknowledgement: Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). The MIT Press.
import numpy as np



# Shared functions

def run_episode(env, Q, epsilon=0.2, gamma=0.95, max_steps=100):
    '''
    Run an episode
    Inputs:
    env: environment
    Q: (num_s, num_a) array of action-value function
    epsilon: for epsilon-greedy policy
    gamma: discount factor
    max_steps: maximum steps in episode

    Returns:
    path: list of (state, action, reward) tuples
    G: episodal discounted return
    '''
    # reset env to new episode, get initial state
    s, _ = env.reset()
    path = []
    # discounted return
    G = 0.0

    for n in range(max_steps):
        a = eps_greedy_pi(Q, s, epsilon)
        # run a step, get next state, reward, terminated, truncated, _ (dont need auxiliary info)
        s_next, r, terminated, truncated, _ = env.step(a)

        path.append((s, a, r))
        G += (gamma**n) * r

        if terminated or truncated:
            break
        s = s_next

    return path, G



def eps_greedy_pi(Q, s, epsilon=0.2):
    '''
    Epsilon-greedy policy for action selection

    Inputs:
    Q: (num_s, num_a) array of action-value function
    s: current state

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
        best = np.flatnonzero(q_s == max_q)
        # action for the state corresponds to highest Q value (w/ tiebreaks)
        a = np.random.choice(best)

    return a

    





def MC_control(env, n_episodes=5000, epsilon=0.1, gamma=0.95, max_steps=1000):
    '''
    '''   

    num_s = env.observation_space.n
    num_a = env.action_space.n

    Q = np.zeros((num_s, num_a))
    count = np.zeros((num_s, num_a))

    for _ in range(n_episodes):
        path, _ = run_episode(env, Q, epsilon=epsilon, gamma=gamma, max_steps=max_steps)
        G = 0.0
        # empty set to store visited states
        visited = set()


        for s, a, r in reversed(path):
            G = r + gamma * G

            if (s, a) in visited:
                continue

            visited.add((s, a))
            count[s, a] += 1
            # incremental mean
            Q[s, a] += (G - Q[s, a]) / count[s, a]
    
    return Q




