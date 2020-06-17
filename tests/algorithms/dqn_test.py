import pytest
import numpy as np
import torch
import random
from tests.fixtures.env import env
from tests.fixtures.net.dqn import net
from agents.algorithms.dqn import Agent, Transition

@pytest.fixture
def agent(env, net):
    return Agent(
        env=env(obs_size=9, n_actions=3), 
        net=net,
        epsilon_decay=10)

@pytest.fixture
def transitions(agent):
    agent.load_replay_buffer(episodes_count=agent.batch_size*2)
    transitions = agent.replay_buffer.sample(agent.batch_size)
    return transitions

def test_eps(agent):
    x = agent.eps(start=1, end=0, decay=200, steps=1)
    assert round(x, 3) == 0.995

    x = agent.eps(start=1, end=0, decay=200, steps=2000)
    assert round(x, 3) == 0

def test_dqn_select_action_random(agent):
    obs = agent.env.reset()
    obs = torch.FloatTensor(obs)
    actions = []
    for step_n in range(1000):
        # should produce highly random actions
        x = agent.select_action(
            policy=agent.policy_net, 
            state=obs, 
            epsilon=True, 
            steps=step_n,
            legal_actions=agent.env.legal_actions)
        actions.append(x)
    # assert that randomness covered all action space
    assert len(np.unique(actions)) == agent.env.action_space_n

def test_dqn_select_action_greedy(agent):
    obs = agent.env.reset()
    obs = torch.FloatTensor(obs)
    actions = []
    for _ in range(100):
        # should produce the same actions since we are passing the same observation
        x = agent.select_action(policy=agent.policy_net, state=obs, epsilon=False, steps=None)
        actions.append(x)
    assert len(np.unique(actions)) == 1

def test_dqn_load_replay_buffer(agent):
    assert len(agent.replay_buffer) == 0
    agent.load_replay_buffer()
    assert len(agent.replay_buffer) > 0

def test_dqn_state_action_values(agent, transitions):
    batch = Transition(*zip(*transitions))
    state_action_values = agent.state_action_values(batch)
    assert len(state_action_values) == agent.batch_size

    # verify that we are calculating the right values
    for i in range(agent.batch_size):
        actions = agent.policy_net(torch.Tensor(batch.state[i]), batch.legal_actions[i])
        x = actions[batch.action[i]]
        y = state_action_values[i].unsqueeze(0)
        # float comparison up to 3 decimal points
        assert abs(x.item()-y.item()) < 1e-3

def test_expected_dqn_state_action_values(env, agent, transitions):
    batch = Transition(*zip(*transitions))
    expected_state_action_values = agent.expected_state_action_values(batch)
    assert len(expected_state_action_values) == agent.batch_size

def test_expected_double_dqn_state_action_values(env, agent, transitions):
    agent.double = True
    batch = Transition(*zip(*transitions))
    expected_state_action_values = agent.expected_state_action_values(batch)
    assert len(expected_state_action_values) == agent.batch_size