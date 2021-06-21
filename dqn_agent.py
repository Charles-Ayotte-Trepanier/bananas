import numpy as np
import random
import os
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 8  # how often to update the network
TD_ERROR_ADJ = 1E-6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
  """Interacts with and learns from the environment."""

  def __init__(self, state_size=37, action_size=4, seed=0):
    """Initialize an Agent object.

    Params
    ======
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        seed (int): random seed
    """
    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.explore_step = 10000
    self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step

    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)

    # Q-Network
    self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
    self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
    self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

    # Replay memory
    self.memory = Memory(BUFFER_SIZE)
    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0

  # save sample (error,<s,a,r,s'>) to the replay memory
  def append_sample(self, state, action, reward, next_state, done):
    state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
    action_tensor = torch.from_numpy(np.array([action])).long().to(device)\
                         .unsqueeze(0)
    next_state_tensor = torch.from_numpy(next_state).float().to(device)\
                             .unsqueeze(0)
    reward_tensor = torch.from_numpy(np.array([reward])).float().to(device)\
                         .unsqueeze(0)
    done_tensor = torch.from_numpy(np.array([int(done)])).float().to(device)\
                       .unsqueeze(0)

    target = self.q_targets(next_state_tensor, reward_tensor, done_tensor)\
                 .cpu().detach().numpy()[0][0]
    pred = self.q_expected(state_tensor, action_tensor)\
               .cpu().detach().numpy()[0][0]

    error = abs(target - pred) + TD_ERROR_ADJ

    self.memory.add(error, (state, action, reward, next_state, done))

  def step(self, state, action, reward, next_state, done):
    # Save experience in replay memory
    self.append_sample(state, action, reward, next_state, done)

    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step + 1) % UPDATE_EVERY
    if self.t_step == 0:
      # If enough samples are available in memory, get random subset and learn
      if self.memory.n_entries > BATCH_SIZE:
        self.learn()

  def save(self, model_folder):
    model_state = self.qnetwork_local.state_dict()
    path = os.sep.join([model_folder, 'checkpoint.pth'])
    torch.save(model_state, path)

  def load(self, model_folder):
    path = os.sep.join([model_folder, 'checkpoint.pth'])
    state_dict = torch.load(path)
    self.qnetwork_local.load_state_dict(state_dict)
    self.qnetwork_target.load_state_dict(state_dict)


  def act(self, state):
    """Returns actions for given state as per current policy.

    Params
    ======
        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection
    """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    self.qnetwork_local.eval()
    with torch.no_grad():
      action_values = self.qnetwork_local(state)
    self.qnetwork_local.train()

    # Epsilon-greedy action selection
    if random.random() > self.epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def q_targets(self, next_states, rewards, dones):
    # Get max predicted Q values (for next states) from target model
    argmaxes = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(-1)

    Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,
                                                                       argmaxes)

    # Compute Q targets for current states
    Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
    return Q_targets

  def q_expected(self, states, actions):
    Q_expected = self.qnetwork_local(states).gather(1, actions)
    return Q_expected

  def learn(self):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
        experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor
    """

    if self.epsilon > self.epsilon_min:
      self.epsilon -= self.epsilon_decay

    (states, actions, rewards, next_states, dones), idxs, is_weights = \
        self.memory.sample(BATCH_SIZE)

    targets = self.q_targets(next_states, rewards, dones)

    # Get expected Q values from local model
    preds = self.q_expected(states, actions)

    errors = torch.abs(preds - targets).cpu().data.numpy()
    # update priority
    for i in range(BATCH_SIZE):
        idx = idxs[i]
        self.memory.update(idx, errors[i])

    # Compute loss
    loss = (is_weights * F.mse_loss(preds, targets)).mean()
    # Minimize the loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # ------------------- update target network ------------------- #
    self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

  def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(),
                                         local_model.parameters()):
      target_param.data.copy_(
        tau * local_param.data + (1.0 - tau) * target_param.data)


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
  e = 0.001
  a = 0.6
  beta = 0.4
  beta_increment_per_sampling = 0.00001

  def __init__(self, capacity):
    self.tree = SumTree(capacity)
    self.capacity = capacity

  def _get_priority(self, error):
    return (np.abs(error) + self.e) ** self.a

  @property
  def n_entries(self):
    return self.tree.n_entries

  def add(self, error, sample):
    p = self._get_priority(error)
    self.tree.add(p, sample)

  def sample(self, n):
    batch = []
    idxs = []
    segment = self.tree.total() / n
    priorities = []

    self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

    for i in range(n):
      a = segment * i
      b = segment * (i + 1)

      #patch - sometimes returns unexisting nodes (data = 0)
      data = 0
      while data == 0:
        s = random.uniform(a, b)
        (idx, p, data) = self.tree.get(s)

      priorities.append(p)
      batch.append(data)
      idxs.append(idx)

    sampling_probabilities = priorities / self.tree.total()
    is_weight = np.power(self.tree.n_entries * sampling_probabilities,
                         -self.beta)
    is_weight /= is_weight.max()
    is_weight = torch.from_numpy(is_weight).float().to(device)

    states = torch.from_numpy(
        np.vstack([e[0] for e in batch if e is not None])).float().to(
        device)
    actions = torch.from_numpy(
        np.vstack([e[1] for e in batch if e is not None])).long().to(
        device)
    rewards = torch.from_numpy(
        np.vstack([e[2] for e in batch if e is not None])).float().to(
        device)
    next_states = torch.from_numpy(np.vstack(
        [e[3] for e in batch if e is not None])).float().to(
        device)
    dones = torch.from_numpy(
        np.vstack([e[4] for e in batch if e is not None]).astype(
            np.uint8)).float().to(device)

    return (states, actions, rewards, next_states, dones), idxs, is_weight

  def update(self, idx, error):
    p = self._get_priority(error)
    self.tree.update(idx, p)


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])