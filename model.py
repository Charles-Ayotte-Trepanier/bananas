import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc = nn.Linear(64, 32)
        self.fc_v = nn.Linear(64, 32)

        self.fc3 = nn.Linear(32, action_size)
        self.fc3_v = nn.Linear(32, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        adv = F.relu(self.fc(x))
        adv = self.fc3(adv)
        advAverage = torch.mean(adv, dim=1, keepdim=True)

        v = F.relu(self.fc_v(x))
        v = self.fc3_v(v)
        return v + (adv - advAverage)
