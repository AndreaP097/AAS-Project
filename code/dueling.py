import torch
import torch.nn.functional as F


class DuelingDQN(torch.nn.Module):

    def __init__(self, state_space, action_space):
        super().__init__()

        self.lin1 = torch.nn.Linear(state_space, 64)
        self.lin2 = torch.nn.Linear(64, 64)

        self.v1 = torch.nn.Linear(64, 64)
        self.V = torch.nn.Linear(64, 1)

        self.a1 = torch.nn.Linear(64, 64)
        self.A = torch.nn.Linear(64, action_space)

    def forward(self, x):

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        v = F.relu(self.v1(x))
        v = self.V(v)

        a = F.relu(self.a1(x))
        a = self.A(a)

        Q = v + (a - a.mean())

        return Q