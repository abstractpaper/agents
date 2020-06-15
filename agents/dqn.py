import torch
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
import random
import math
import copy
from collections import namedtuple
from itertools import count
from tensorboardX import SummaryWriter

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'legal_actions'))

class Agent:
    def __init__(self, env, net, double=False, dev=None):
        global device
        device = dev

        self.double = double            # double q learning
        # self.episodes_count = 400000
        self.loss_cutoff = 0.1
        self.learning_rate = 3e-5       # alpha
        self.batch_size = 128
        self.epsilon_start = 1          # start with 100% exploration
        self.epsilon_end = 0.1          # end with 10% exploration
        self.epsilon_decay = 30000      # higher value = slower decay
        self.discount = 1               # gamma
        self.target_update = 1000       # number of steps to update target network
        self.eval_episodes_count = 1000 # number of episodes for evaluation
        self.replay_buffer = ReplayBuffer(1000000)
        self.panic_buffer = PanicBuffer(10000)
        self.panic_value = -1
        self.euphoria_value = 1
        self.env = env
        self.env_clone = copy.deepcopy(env) # separate environment so we don't mess with `env` state; used for buffer loading and evaluation
        self.policy_net = net(self.env.observation_space_n, self.env.action_space_n).to(device) # what drives current actions; uses epsilon.
        self.target_net = net(self.env.observation_space_n, self.env.action_space_n).to(device) # copied from policy net periodically; greedy.

        # init target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def train(self):
        writer = SummaryWriter(comment="-tictactoe")

        # fill replay buffer with some random episodes
        self.load_replay_buffer(episodes_count=self.batch_size*2)

        steps = 1
        recent_loss = []
        lost_episodes_count = 0
        while True:
            # fill replay buffer with one episode from the current policy (epsilon is used)
            self.load_replay_buffer(policy=self.policy_net, steps=steps)

            # optimize policy_net
            loss = self.optimize(self.replay_buffer)
            recent_loss.append(loss)
            recent_loss = recent_loss[-100:]

            # tensorboard metrics
            epsilon = Agent.eps(self.epsilon_start, self.epsilon_end, self.epsilon_decay, steps)
            writer.add_scalar("env/epsilon", epsilon, steps)
            writer.add_scalar("env/replay_buffer", len(self.replay_buffer), steps)
            writer.add_scalar("train/loss", loss, steps)

            # stop if average loss < loss_cutoff
            # and last evaluation has no lost episodes
            loss_achieved = sum(recent_loss)/len(recent_loss) < self.loss_cutoff
            if loss_achieved and lost_episodes_count == 0:
                break

            # update the target network, copying all weights and biases in policy_net to target_net
            if steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())                
                avg_rewards, transitions = self.evaluate_policy(self.target_net)

                lost_episodes_count = len(list(filter(lambda t: t.reward < self.panic_value, transitions)))

                writer.add_scalar("train/avg_rewards", avg_rewards, steps)
                writer.add_scalar("train/lost_episodes_count", lost_episodes_count, steps)
                writer.add_scalar("env/panic_buffer", len(self.panic_buffer), steps)

                if type(self.panic_buffer) == PanicBuffer:
                    # drain panic buffer and optimize
                    n = 0
                    while len(self.panic_buffer) > self.batch_size:
                        loss = self.optimize(self.panic_buffer)
                        writer.add_scalar("train/panic_loss", loss, steps+n)
                        n = n+1

            steps = steps + 1

        # save model
        torch.save(self.target_net.state_dict(), "policies/dqn")
        writer.close()

    @staticmethod
    def eps(start, end, decay, steps): 
        # compute epsilon threshold
        return end + (start - end) * math.exp(-1. * steps / decay)

    def select_action(self, policy, state, epsilon=False, steps=None, legal_actions=[]):
        """ 
        selects an action with a chance of being random if epsilon is True,
        otherwise selects the action produced by policy.
        """
        if epsilon:
            if steps == None:
                raise ValueError(f"steps must be an integer. Got = {steps}")

            # pick a random number
            sample = random.random()
            # see what the dice rolls
            threshold = Agent.eps(self.epsilon_start, self.epsilon_end, self.epsilon_decay, steps)
            if sample <= threshold:
                # explore
                action = random.choice([i for i in range(self.env.action_space_n+1) if i in legal_actions])
                return torch.tensor([[action]], device=device, dtype=torch.long)
        
        # greedy action
        with torch.no_grad():
            # index of highest value item returned from policy -> action
            return policy(state, legal_actions).argmax().view(1, 1)

    def optimize(self, buffer):
        if len(buffer) < self.batch_size:
            return

        # sample from replay buffer
        transitions = buffer.sample(self.batch_size)
        # n transitions -> 1 transition with each attribute containing all the
        # data point values along its axis.
        # e.g. batch.action = list of all actions from each row
        batch = Transition(*zip(*transitions))

        # compute state action values (what policy_net deems as the best action)
        state_action_values = self.state_action_values(batch)
        # compute expected state action values (reward + value of next state according to target_net)
        expected_state_action_values = self.expected_state_action_values(batch)

        # calculate difference between actual and expected action values
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # adam optimizer
        optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        # calculate gradients
        loss.backward()
        for param in self.policy_net.parameters():
            # clip gradients
            param.grad.data.clamp_(-1, 1)
        # optimize policy_net
        optimizer.step()

        return loss

    def load_replay_buffer(self, policy=None, episodes_count=1, steps=0):
        """ load replay buffer with episodes_count """
        env = self.env_clone
        for eps_idx in range(episodes_count):
            state = env.reset()
            while True:
                legal_actions = env.legal_actions
                if not policy:
                    action = random.choice([i for i in range(self.env.action_space_n+1) if i in legal_actions])
                else:
                    action = self.select_action(
                        policy=policy, 
                        state=torch.Tensor(state).to(device), 
                        epsilon=True,
                        steps=steps,
                        legal_actions=legal_actions).item()

                # perform action
                next_state, reward, done, _ = env.step(action)
                reward = torch.tensor([reward], device=device)

                # insert into replay buffer
                action = torch.tensor([[action]], device=device, dtype=torch.long)
                self.replay_buffer.push(state, action, next_state if not done else None, reward, legal_actions)

                if done:
                    break
                else:
                    # transition
                    state = next_state

    def evaluate_policy(self, policy):
        env = self.env_clone
        rewards = []
        transitions = []
        for _ in range(self.eval_episodes_count):
            state = env.reset()
            while True:
                legal_actions = env.legal_actions
                action = self.select_action(
                    policy=policy, 
                    state=torch.Tensor(state).to(device), 
                    epsilon=False,
                    legal_actions=legal_actions).item()
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)

                action = torch.tensor([[action]], device=device, dtype=torch.long)
                transitions.append(Transition(state, action, next_state if not done else None, reward, legal_actions))

                if done:
                    break
                else:
                    state = next_state

        if type(self.panic_buffer) == PanicBuffer:
            # fill panic buffer
            panic_generator = filter(lambda t: t.reward < self.panic_value, transitions)
            for t in panic_generator:
                self.panic_buffer.push(*t)
            panic_generator = filter(lambda t: t.reward > self.euphoria_value, transitions)
            for t in panic_generator:
                self.panic_buffer.push(*t)

        avg_rewards = sum(rewards)/self.eval_episodes_count if self.eval_episodes_count > 0 else 0
        return avg_rewards, transitions

    def state_action_values(self, batch):
        """ 
        Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        columns of actions taken. These are the actions which would've been taken
        for each batch state according to policy_net.
        """
        # all states in batch
        state_batch = torch.Tensor(batch.state).to(device)
        # get action values for each state in batch
        state_action_values = self.policy_net(state_batch, batch.legal_actions)
        # all actions in batch
        action_batch = torch.cat(batch.action)
        # select action from state_action_values according to action_batch value
        return state_action_values.gather(1, action_batch)

    def expected_state_action_values(self, batch):
        """
        Compute V(s_{t+1}) for all next states.
        Expected values of actions for non_final_next_states are computed based
        on the "older" target_net; selecting their best reward with max(1)[0].
        This is merged based on the mask, such that we'll have either the expected
        state value or 0 in case the state was final.
        """
        # a list indicating whether each state is a non-final state.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.Tensor([s for s in batch.next_state if s is not None]).to(device)
        reward_batch = torch.Tensor([[r] for r in batch.reward]).to(device)

        next_legal_actions = [i for (i, v) in zip(list(batch.legal_actions), non_final_mask.tolist()) if v]
        next_state_values = torch.zeros(self.batch_size, device=device)

        if len(non_final_next_states) > 0:
            if self.double:
                # double q learning: get actions from policy_net and get their values according to target_net
                # Q(st+1, a)
                next_state_actions = self.policy_net(non_final_next_states, next_legal_actions).max(1)[1].unsqueeze(-1)
                # max Q`(st+1, max Q(st+1, a) )
                next_state_values[non_final_mask] = self.target_net(non_final_next_states, next_legal_actions).gather(1, next_state_actions).squeeze(-1)
            else:
                # max Q`(st+1, a)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states, next_legal_actions).max(1)[0].detach()

        # Compute the expected Q values
        # reward + max Q`(st+1, a) * discount
        state_action_values = reward_batch + (next_state_values.unsqueeze(1) * self.discount)

        return state_action_values
        

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PanicBuffer(ReplayBuffer):
    """ 
    PanicBuffer is ReplayBuffer that removes sampled items from memory.
    One use case is to store surprising results that have a large loss,
    sample them and optimize during evaluation.
    """
    def sample(self, batch_size):
        """ get a sample and remove items from memory """
        if len(self.memory) < batch_size:
            return None
        sample = [self.memory.pop(random.randrange(len(self.memory))) for _ in range(batch_size)]
        self.position = len(self.memory)
        return sample
