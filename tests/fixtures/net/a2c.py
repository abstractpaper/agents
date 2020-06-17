import pytest
import torch.nn as nn 
import torch.nn.functional as F
from net.feed_forward import FeedForward

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
        # observations -> hidden layer with relu activation
        common = nn.Sequential(
            nn.Linear(self.obs_size, self.hidden_layer),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(self.hidden_layer, self.n_actions)
        self.critic_head = nn.Linear(self.hidden_layer, 1)
        return common

    def forward(self, x, legal_actions):
        # shared layers among actor and critic
        common = self.net(x)

        # actor layer
        if len(legal_actions) > 0:
            actions = self.mask_actions(self.actor_head(common), legal_actions)
        else:
            actions = self.actor_head(common)
        action_dist = F.softmax(actions, dim=-1)

        # critic layer
        value = self.critic_head(common)

        return action_dist, value