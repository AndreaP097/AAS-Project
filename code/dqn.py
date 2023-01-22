import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DQN(nn.Module):
    """
    Create the Deep-Q-Network.
    """
    def __init__(self, state_space, action_space):
        super().__init__()

        self.lin1 = nn.Linear(
            in_features = state_space, 
            out_features = 64
        )
        self.lin2 = nn.Linear(
            in_features = 64,
            out_features = 64
        )
        self.lin3 = nn.Linear(
            in_features = 64 ,
            out_features = action_space
        )

    
    def forward(self, x):

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x