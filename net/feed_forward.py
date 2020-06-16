import abc
import torch.nn as nn 
import math

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.net = self.model()

    @abc.abstractmethod
    def model(self):
        """
        Return a feed forward network model.
        """

    def forward(self, x, legal_actions=[]):
        # forward pass
        actions = self.net(x)
        # mask actions if a mask is provided
        if len(legal_actions) > 0:
            actions = self.mask_actions(actions, legal_actions)
        return actions

    def mask_actions(self, actions, mask):
        if actions.dim() == 2:
            # batch
            if len(actions) != len(mask):
                raise Exception("actions and mask batches have different sizes.")

            for idx, step in enumerate(actions):
                illegal_actions = [i for i, _ in enumerate(step) if i not in mask[idx]]
                for jdx in illegal_actions:
                    actions[idx][jdx] = -math.inf
        else:
            # single step
            illegal_actions = [i for i, _ in enumerate(actions) if i not in mask]
            for idx in illegal_actions:
                actions[idx] = -math.inf

        return actions