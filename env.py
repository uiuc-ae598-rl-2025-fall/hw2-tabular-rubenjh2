# Acknowledgement to https://gymnasium.farama.org/introduction/basic_usage/
# and https://gymnasium.farama.org/environments/toy_text/frozen_lake/ for setup and environment
import numpy as np
import gymnasium as gym


def build_env(success_rate=1.0/3.0, is_slippery=True):
    '''
    Build FrozenLake environment with desired map, success rate, and slipperiness.

    Inputs:
    gamma: discount factor (not used in env but for reference)
    success_rate: probability of intended action being taken
    is_slippery: if True, the environment is stochastic
    '''
    env = gym.make(
            'FrozenLake-v1',
            desc=["SFFF", "FHFH", "FFFH", "HFFG"],
            map_name="4x4",
            is_slippery=is_slippery,
            success_rate=success_rate,
            # (reach goal, reach hole, reach frozen)
            reward_schedule=(1, 0, 0)
            )
    
    return env


    

    