import pytest
import torch.nn as nn 
from agents.net.feed_forward import FeedForward

@pytest.fixture
def net():
    return Net

class Net(FeedForward):
    def __init__(self, obs_size, n_actions, hidden_layer=32):
        # model is initiated in parent class, set params early.
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.hidden_layer = hidden_layer
        super(Net, self).__init__()

    def model(self):
        # observations -> hidden layer with relu activation -> actions
        return nn.Sequential(
            nn.Linear(self.obs_size, self.hidden_layer),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, self.n_actions)
        )