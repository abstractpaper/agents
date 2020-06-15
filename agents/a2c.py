# Advantage Actor Critic (A2C)
# ============================
# Actor:  Optimize a neural network to produce a probability distribution of actions.
# Critic: Optimize a neural network to produce V(s) which allows us to compute the
#         advantage of an action-state pair A(s,a). A(s,a) is used to scale gradients
#         computed by the actor, hence acting as a critic. Scaling gradients in this
#         way allows the critic to reinforce actor actions.

import torch
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
import random
import math
import copy
from torch import nn
from collections import namedtuple
from itertools import count
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Agent:
    def __init__(self, env, net, dev=None):
        global device
        device = dev

        self.learning_rate = 3e-4       # alpha
        self.discount = 0.9             # gamma
        self.eval_episodes_count = 100  # number of episodes for evaluation
        self.env = env
        self.env_clone = copy.deepcopy(env) # separate environment so we don't mess with `env` state; used for evaluation
        self.net = net(self.env.observation_space_n, self.env.action_space_n).to(device)

    def train(self):
        writer = SummaryWriter(comment="-tictactoe")
        
        ep_idx = 1
        running_reward = 0

        # infinite episodes until threshold is met off
        while True:
            state = self.env.reset()
            ep_reward = 0
            step_rewards = []
            saved_actions = []
            entropy = 0

            # a single episode is finite
            while True:
                legal_actions = self.env.legal_actions

                # choose an action
                action, action_dist = self.select_action(state, legal_actions, saved_actions)
                # take a step in env
                next_state, reward, done, _ = self.env.step(action)
                
                # calculate entropy
                entropy += -torch.sum(torch.mean(action_dist) * torch.log(action_dist))
                # rewards
                step_rewards.append(reward)
                ep_reward += reward

                if done:
                    break

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            # optimize policy_net
            loss = self.optimize(step_rewards, saved_actions, entropy)

            # tensorboard metrics
            writer.add_scalar("train/loss", loss, ep_idx)
            writer.add_scalar("train/running_reward", running_reward, ep_idx)

            ep_idx = ep_idx + 1
            # # stop if average loss < loss_cutoff
            # if sum(recent_loss)/len(recent_loss) < self.loss_cutoff:
            #     break

            if ep_idx % 1000 == 0:
                eval_avg_rewards = self.evaluate_policy()
                writer.add_scalar("train/eval_avg_rewards", eval_avg_rewards, ep_idx)

        # save model
        torch.save(self.target_net.state_dict(), "policies/a2c")
        writer.close()

    def select_action(self, state, legal_actions, saved_actions):
        action_dist, value = self.net(torch.Tensor(state).to(device), legal_actions)

        print(action_dist)

        m = Categorical(action_dist)
        action = m.sample()

        saved_actions.append(SavedAction(m.log_prob(action), value))

        return action.item(), action_dist

    def optimize(self, step_rewards, saved_actions, entropy):
        """
        Calculates actor and critic loss and performs backprop.
        """
        R = 0
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in step_rewards[::-1]:
            # calculate the discounted value
            R = r + R * self.discount
            returns.insert(0, R)

        # smallest number such that 1.0 + eps != 1.0
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        # normalize returns; use one scale for values
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss.
            # we are scaling the actor's action probability by advantage;
            # advantage can be positive or negative.
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() # + 0.001 * entropy

        # reset gradients
        optimizer = optim.Adam(params=self.net.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        # perform backprop; compute gradient
        loss.backward()
        # clip gradients
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        # update net parameters
        optimizer.step()

        return loss

    def evaluate_policy(self):
        env = self.env_clone
        rewards = []
        ep_rewards = []
        for _ in range(self.eval_episodes_count):
            state = env.reset()
            ep_reward = 0
            while True:
                legal_actions = env.legal_actions
                action_dist, value = self.net(torch.Tensor(state).to(device), legal_actions)
                action = torch.argmax(action_dist).item()
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                ep_reward += reward

                if done:
                    break
                else:
                    state = next_state
            ep_rewards.append(ep_reward)

        return sum(rewards)/self.eval_episodes_count if self.eval_episodes_count > 0 else 0