import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch import optim
import numpy as np
import os
import yaml

configs = yaml.safe_load(open("config.yaml"))

# Actor
class PolicyNetwork(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(PolicyNetwork, self).__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        
        if configs["env"]["tron"]:
            modelname = "tron"
        else:
            modelname = configs["env"]["name"]
        self.checkpoint_file = os.path.join("models", modelname,"actor_checkpoint")
        self.hidden_units = configs["model"]["hidden_units"]
        self.learning_rate = configs["model"]["learning_rate"]
        
        self.actor_seq = nn.Sequential(
            nn.Linear(*self.num_observations, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.num_actions),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
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
    def __init__(self, num_observations):
        super(ValueNetwork, self).__init__()
        self.num_observations = num_observations
        
        if configs["env"]["tron"]:
            modelname = "tron"
        else:
            modelname = configs["env"]["name"]
        self.checkpoint_file = os.path.join("models", modelname,"critic_checkpoint")
        self.hidden_units = configs["model"]["hidden_units"]
        self.learning_rate = configs["model"]["learning_rate"]
        
        self.critic_seq = nn.Sequential(
            nn.Linear(*self.num_observations, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        x = self.critic_seq(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))