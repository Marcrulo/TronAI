import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch import optim
import numpy as np
import os

# Actor
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim):
        super(PolicyNetwork, self).__init__()
        self.input_dim = input_dim
        
        self.checkpoint_file = os.path.join("models", 'actor_checkpoint')
        
        self.actor_seq = nn.Sequential(
            nn.Linear(*self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        x = self.actor_seq(x)
        x = Categorical(x)
        return x
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    

# Critic
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join("models", 'critic_checkpoint')
        self.input_dim = input_dim
        self.critic_seq = nn.Sequential(
            nn.Linear(*self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        x = self.critic_seq(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))