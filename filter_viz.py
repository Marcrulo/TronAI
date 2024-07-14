# Imports
from tron_env import TronEnv
from matplotlib import pyplot as plt
import os
from agent import Agent
import gym
import numpy as np
import pandas as pd
from torch.utils import tensorboard
import torch
import yaml

# Load the configuration file
configs = yaml.safe_load(open("config.yaml"))
cnn = configs["env"]["cnn"]

# Create the environment
if configs["env"]["name"] == "tron":
    env = TronEnv(render_mode=configs["env"]["render_mode"], cnn=cnn)
else:
    env = gym.make(configs["env"]["name"], render_mode=configs["env"]["render_mode"], cnn=cnn)
_ = env.reset()

# Create the agents
if cnn:
    obs_space = env.observation.shape
else:
    obs_space = env.observation_space.shape
    
player1 = Agent(num_actions=env.action_space.n, 
                num_observations=obs_space,
                cnn=cnn)
player2 = Agent(num_actions=env.action_space.n,
                num_observations=obs_space,
                cnn=cnn)
player1.load_models()

from torchvision import utils
model = player1.actor
kernel = model.actor_seq[0].weight.data.cpu().numpy()
print(kernel.shape)

filter_img = utils.make_grid(torch.tensor(kernel), nrow=4, normalize=True)
plt.imshow(filter_img.permute(1, 2, 0))
plt.show()