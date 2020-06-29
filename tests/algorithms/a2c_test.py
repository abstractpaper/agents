import pytest
import numpy as np
import torch
import random
import gym
from gym.envs.registration import register, registry
from tests.fixtures.net.a2c import net
from agents.algorithms.a2c import Agent

@pytest.fixture
def agent(net):
    env_id = 'TestEnv-v0'

    if env_id not in [spec.id for spec in registry.all()]:
        register(
            id=env_id,
            entry_point='tests.fixtures.env:Env',
            reward_threshold=1.5,
        )
        
    env = gym.make(env_id, obs_size=9, n_actions=3)

    return Agent(
        env=env, 
        net=net)

def test_run_episode(agent):
    ep_reward, step_rewards, saved_actions, entropy = agent.run_episode()
    assert isinstance(ep_reward, np.integer)
    assert isinstance(step_rewards, list) and len(step_rewards) > 0
    assert ep_reward == sum(step_rewards)
    assert isinstance(saved_actions, list)
    assert isinstance(entropy, torch.Tensor)

def test_evaluate_policy(agent):
    running_reward = 0
    stop, avg_rewards = agent.evaluate_policy(running_reward)

def test_select_action(agent):
    obs = agent.env.reset()
    obs = torch.FloatTensor(obs)
    saved_actions = []
    n_tests = 1000
    for step_n in range(n_tests):
        action, action_dist, dist_entropy = agent.select_action(
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

def test_calculate_returns(agent):
    step_rewards = [-0.1, -0.1, 1]
    rewards = agent.calculate_returns(step_rewards)

    r2 = step_rewards[-1] + 0 * agent.discount
    r1 = step_rewards[-2] + r2 * agent.discount
    r0 = step_rewards[-3] + r1 * agent.discount

    assert rewards == [r0, r1, r2]

def test_standardize_returns(agent):
    returns = [0.7811, 0.89, 1.0]

    std_returns = agent.standardize_returns(returns)

    # mean is 0 (+/- fractions)
    assert abs(std_returns.mean() - 0) < 1e-3
    # std is 1 (+/- fractions)
    assert abs(std_returns.std() - 1.0) < 1e-3