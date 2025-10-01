import env
import numpy as np

def main():

    environment = env.build_env(gamma=0.95, success_rate=1.0/3.0, is_slippery=True)

    s = environment.observation_space
    a = environment.action_space

    num_s = s.n
    num_a = a.n

    Q = np.zeros([num_s, num_a])
    print(np.shape(Q))

    print(s, num_s)
    print(a, num_a)
    print(Q[5])




if __name__ == "__main__":
    main()