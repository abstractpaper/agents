import pytest
import numpy as np
import torch
import random
from collections import namedtuple
from prop.buffers.priority_replay_buffer import PrioritizedReplayBuffer

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'mask'))

@pytest.fixture
def buffer():
    buffer = PrioritizedReplayBuffer(capacity=1000)
    return buffer

def test_push(buffer):
    assert buffer.tree.n_entries == 0
    for _ in range(10):
        buffer.push(0.5, Transition(np.zeros(9), [1], None, 1, np.zeros(3)))
    assert buffer.tree.n_entries == 10

def test_sample(buffer):
    # add 1000 samples with error (priority) increasing for each n
    for n in range(1000):
        sample = Transition(np.zeros(9), [n], None, n, np.zeros(3))
        buffer.push(n/1000, sample)

    def normalize(a):
        max_n = max(a)
        return [v/max_n for v in a]

    results = []
    for _ in range(100):
        batch, _, _ = buffer.sample(32)

        # reward=n
        td = sorted([b.reward for b in batch])
        # normalize
        td_normalized = np.array(normalize(td))

        # sample from an exponential curve
        exp_curve = sorted(np.random.exponential(size=32))
        # normalize
        exp_curve_normalized = np.array(normalize(exp_curve))

        # calculate MSE for both lists
        mse = ((td_normalized - exp_curve_normalized)**2).mean(axis=0)
        results.append(mse)

    # assert that the two curves don't deviate much; meaning that our buffer
    # is sampling transitions with higher priorities more often.
    assert max(results) < 0.5
    
def test_update(buffer):
    for n in range(10):
        buffer.push(n/10, Transition(np.zeros(9), [n], None, n, np.zeros(3)))
    buffer.update(1000, 0.9)
    # assert new priority = (error + epsilon) ^ alpha
    assert buffer.tree.tree[1000] == (np.abs(0.9) + buffer.e) ** buffer.alpha