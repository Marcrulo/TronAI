import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch import optim
import numpy as np
import os
import yaml

from matplotlib import pyplot as plt

configs = yaml.safe_load(open("config.yaml"))

# Actor
class PolicyNetwork(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(PolicyNetwork, self).__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions

        modelname = configs["env"]["name"]
        self.dir = os.path.join("models", modelname)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.checkpoint_file = os.path.join(self.dir,"actor_checkpoint")
        self.hidden_units = configs["model"]["hidden_units"]
        self.learning_rate = configs["model"]["learning_rate"]
        
        # game
        scale = 1#configs['game']['scale']

        self.actor_seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=scale*3, stride=1, padding=0),
            nn.ReLU(),
            # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=scale*3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(14400,self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.num_actions),
            nn.Softmax(dim=-1)
        )

        self._initialize_weights()
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        # print(x.shape)
        # x = torch.zeros(1,17,17).to(self.device)
        
        # show x
        print(x.cpu().detach().numpy().shape)
        plt.imshow(x.cpu().detach().numpy()[0], cmap='jet', vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()
        x = x.unsqueeze(1)
        x = self.actor_seq(x)
        print(np.round(x.cpu().detach().numpy(),2))        
        x = Categorical(x)
        return x
    
    def _initialize_weights(self):
        final_linear_layer = self.actor_seq[-2]
        
        # Xavier initialization
        # nn.init.xavier_normal_(final_linear_layer.weight)
        # nn.init.constant_(final_linear_layer.bias, 0)
        
        # He initialization
        nn.init.kaiming_normal_(final_linear_layer.weight)
        nn.init.constant_(final_linear_layer.bias, 0)
        
        # nn.init.normal_(final_linear_layer.weight, mean=0, std=1)
        # nn.init.constant_(final_linear_layer.bias, 0)
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    

# Critic
class ValueNetwork(nn.Module):
    def __init__(self, num_observations):
        super(ValueNetwork, self).__init__()
        self.num_observations = num_observations

        modelname = configs["env"]["name"]
        self.dir = os.path.join("models", modelname)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.checkpoint_file = os.path.join(self.dir,"critic_checkpoint")
        self.hidden_units = configs["model"]["hidden_units"]
        self.learning_rate = configs["model"]["learning_rate"]
        
        # game
        scale = 1 #configs['game']['scale']

        self.critic_seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=scale*3, stride=1, padding=0),
            nn.ReLU(),
            # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=scale*3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(14400,self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, 1),
        )

        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.critic_seq(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))