# Acknowledgement to https://gymnasium.farama.org/introduction/basic_usage/
# and https://gymnasium.farama.org/environments/toy_text/frozen_lake/ for setup and environment
import numpy as np
import gymnasium as gym


def build_env(gamma=0.95, success_rate=1.0/3.0, is_slippery=True):
    '''
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


    

    