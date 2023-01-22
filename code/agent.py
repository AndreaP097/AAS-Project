import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from dqn import DQN
from dueling import DuelingDQN
from experience_replay import experience_replay
import os

class Agent:
    """
    Initialize the agent.
    """

    def __init__(
        self,
        model,
        state_space,
        action_space,
        buffer_size,
        batch_size, 
        learning_rate,
        gamma, 
        tau,
        device):

        self.model_name = model
        self.action_space = action_space
        self.device = device

        # Hyperparameters 
        self.eps = 1.
        self.eps_min = 0.01
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau

        # Experience replay buffer
        self.buffer = experience_replay(buffer_size, device)

        # Main network and target network
        if self.model_name == 'dqn':
            print("Using DQN model.")
            self.model = DQN(state_space, action_space).to(self.device)
            self.model_target = DQN(state_space, action_space).to(self.device)
        elif self.model_name == 'dueling':
            print("Using DuelingDQN model.")
            self.model = DuelingDQN(state_space, action_space).to(device)
            self.model_target = DuelingDQN(state_space, action_space).to(device)
        else:
            raise ValueError("Select a valid model.")

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)


    def action(self, state):
        """
        Implement the eps-greedy policy for action selection.
        """

        if np.random.rand() <= self.eps:
            action = np.random.choice(range(self.action_space))
        else:
            with torch.no_grad():
                state = torch.tensor(state).unsqueeze(0).to(self.device)
                action = np.argmax(self.model(state).cpu().numpy())

        return action


    def train(self):
        """
        Train the agent for one epoch on a batch sampled from the experience replay buffer.
        """

        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.faster_sample(self.batch_size)
        
        # Compute Q values from main network
        q_pred = self.model(states).gather(1, actions)

        if self.model_name == 'dqn':
            ##### ----- DQN algorithm ----- #####

            # Compute Q values of the next state with the target network
            with torch.no_grad():
                q_next = self.model_target(next_states).max(1)[0].unsqueeze(1)

            # Compute the target Q values using the Bellman equation
            q_targets = rewards + self.gamma * q_next * (1 - dones)

        else:
            ##### ----- Double DQN algorithm ----- #####

            with torch.no_grad():
                # Computation of argmax Q(s', a; θ), this uses the online network
                q_online = self.model(next_states)
                actions_q_argmax = q_online.max(1)[1].unsqueeze(1)

                # Computation of Q(s', argmax Q(s', a; θ); θ'), this uses the target network
                q_next = self.model_target(next_states).gather(1, actions_q_argmax)

            q_targets = rewards + self.gamma * q_next * (1 - dones)


        loss = torch.nn.functional.smooth_l1_loss(q_pred, q_targets)
        # loss = torch.nn.functional.mse_loss(q_pred, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 

    def soft_copy(self):
        """
        Perform a soft copy of the main model into the target model.
        """

        for target_param, model_param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau*model_param.data + (1.0-self.tau)*target_param.data)


    def update_eps(self):
        """
        Update epsilon for the epsilon-greedy policy.
        """

        if self.eps > self.eps_min:
            self.eps = self.eps * 0.995

    def save_model(self, filename):
        
        if self.model_name == 'dqn':
            path = os.path.join('./checkpoints/dqn', filename)
        else:
            path = os.path.join('./checkpoints/dueling', filename)
        
        torch.save(self.model.state_dict(), path)


    def load_model(self, filename):

        if self.model_name == 'dqn':
            path = os.path.join('./checkpoints/dqn', filename)
        else:
            path = os.path.join('./checkpoints/dueling', filename)

        self.model.load_state_dict(torch.load(path))
