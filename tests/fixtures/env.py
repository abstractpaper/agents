import pytest
import numpy as np
import gym
from gym import spaces

@pytest.fixture
def env():
    return Env

class Env(gym.Env):
    environment_name = "Test Environment"

    def __init__(self, obs_size, n_actions):
        # spaces
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Discrete(obs_size)
        # env state
        self._state = [0 for _ in range(obs_size)]
        self._done = False

    def step(self, action):
        if self._done:
            return self.reset()

        # mark a random state index with a random state value; mimicking a real environment state change
        random_idx = np.random.choice(a=self.observation_space_n)
        self._state[random_idx] = np.random.choice(a=5) # an arbitrary number of state values; e.g. 5

        # mark as done 20% of the time
        self._done = np.random.choice(
            a=[True,False], 
            p=[0.2, 0.8]
        )

        reward = np.random.choice(
            a=[0,1], 
            p=[0.6, 0.4]
        )
            
        return (self.state, reward, self._done, dict())

    def reset(self):
        self._done = False
        return self.state

    @property
    def legal_actions(self):
        return list(range(self.action_space_n))

    @property
    def observation_space_n(self):
        return self.observation_space.n

    @property
    def action_space_n(self):
        return self.action_space.n

    @property
    def state(self):
        return self._state
