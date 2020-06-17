import pytest
import numpy as np
import torch
import random
from tests.fixtures.env import env
from tests.fixtures.net.a2c import net
from a2c import Agent

@pytest.fixture
def agent(env, net):
    return Agent(
        env=env(obs_size=9, n_actions=3), 
        net=net)

def test_a2c_select_action(agent):
    obs = agent.env.reset()
    obs = torch.FloatTensor(obs)
    saved_actions = []
    n_tests = 1000
    for step_n in range(n_tests):
        action, action_dist = agent.select_action(
            state=obs,
            legal_actions=agent.env.legal_actions,
            saved_actions=saved_actions)
        assert action in range(agent.env.action_space_n)
        assert len(action_dist) == agent.env.action_space_n
        # action distribution sums up to 1 (up to 1e-3 accuracy)
        assert abs(sum(action_dist) - 1) < 1e-3
        # all probabilities >= 0
        assert [n>=0 for n in action_dist]
    assert len(saved_actions) == n_tests