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
    def __init__(self, num_observations, num_actions, cnn=False):
        super(PolicyNetwork, self).__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.cnn  = cnn

        modelname = configs["env"]["name"]
        self.dir = os.path.join("models", modelname)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.checkpoint_file = os.path.join(self.dir,"actor_checkpoint")
        self.hidden_units = configs["model"]["hidden_units"]
        self.learning_rate = configs["model"]["learning_rate"]
        
        # game
        scale = 1#configs['game']['scale']

        if self.cnn:
            self.actor_seq = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=scale*3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=scale*3, stride=1, padding=1),
                nn.Flatten(),
                nn.Linear(8192,self.hidden_units),
                nn.Tanh(),
                nn.Linear(self.hidden_units, self.num_actions),
                nn.Softmax(dim=-1)
            )
        else:
            self.actor_seq = nn.Sequential(
                nn.Linear(*self.num_observations, self.hidden_units),
                nn.ReLU(),
                nn.Linear(self.hidden_units, self.hidden_units),
                # nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(self.hidden_units, self.num_actions),
                nn.Softmax(dim=-1)
            )
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        x = np.expand_dims(x, axis=1)
        x /= 2
        x = torch.tensor(x).to(self.device)
        x = self.actor_seq(x)
        x = Categorical(x)
        return x
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    

# Critic
class ValueNetwork(nn.Module):
    def __init__(self, num_observations, cnn=False):
        super(ValueNetwork, self).__init__()
        self.num_observations = num_observations
        self.cnn = cnn

        modelname = configs["env"]["name"]
        self.dir = os.path.join("models", modelname)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.checkpoint_file = os.path.join(self.dir,"critic_checkpoint")
        self.hidden_units = configs["model"]["hidden_units"]
        self.learning_rate = configs["model"]["learning_rate"]
        
        # game
        scale = 1#configs['game']['scale']

        if self.cnn:
            self.critic_seq = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=scale*3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=scale*3, stride=1, padding=1),
                nn.Flatten(),
                nn.Linear(8192,self.hidden_units),
                # nn.Dropout(0.2),
                nn.Tanh(),
                nn.Linear(self.hidden_units, 1),
            )
        else:        
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
        x = np.expand_dims(x, axis=1)
        x /= 2
        x = torch.tensor(x).to(self.device)
        x = self.critic_seq(x)
        
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))