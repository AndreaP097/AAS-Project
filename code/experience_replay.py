import torch
import numpy as np
from collections import deque, namedtuple
import random

class experience_replay:

    def __init__(self, buffer_size, device):
        self.buffer = deque(maxlen = buffer_size)
        self.device = device
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    
    def __len__(self):
        return len(self.buffer)


    def add_element(self, state, action, reward, next_state, done: bool):
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)


    def sample_batch(self, batch_size):

        # sample #batch_size indeces 
        idx = np.random.choice(
            a = len(self.buffer),
            size = batch_size,
            replace = False
        )

        batch = np.array(self.buffer, dtype='object')[idx.astype(int)]

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for experience in range(batch_size):
            state, action, reward, next_state, done = batch[experience]
            states.append(torch.tensor(state))
            actions.append(torch.tensor(action))
            rewards.append(torch.tensor(reward))
            next_states.append(torch.tensor(next_state))
            dones.append(done)

        states = torch.stack(states).float().to(self.device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(1).float().to(self.device)
        next_states = torch.stack(next_states).float().to(self.device)
        dones = torch.tensor(dones, dtype=torch.uint8).float().unsqueeze(1).to(self.device)

        return (states, actions, rewards, next_states, dones)


    def faster_sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)

        states = torch.stack([torch.tensor(e.state) for e in experiences if e is not None]).float().to(self.device)
        actions = torch.stack([torch.tensor(e.action) for e in experiences if e is not None]).long().unsqueeze(1).to(self.device)
        rewards = torch.stack([torch.tensor(e.reward) for e in experiences if e is not None]).float().unsqueeze(1).to(self.device)
        next_states = torch.stack([torch.tensor(e.next_state) for e in experiences if e is not None]).float().to(self.device)
        dones = torch.stack([torch.tensor(e.done, dtype=torch.uint8) for e in experiences if e is not None]).unsqueeze(1).to(self.device)
  
        return (states, actions, rewards, next_states, dones)