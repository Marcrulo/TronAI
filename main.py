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
import time 

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
    obs_space = (obs_space[0]-1, obs_space[1]-1)
else:
    obs_space = env.observation_space.shape
    
player1 = Agent(num_actions=env.action_space.n, 
                num_observations=obs_space,
                cnn=cnn)
player2 = Agent(num_actions=env.action_space.n,
                num_observations=obs_space,
                cnn=cnn)
# player1.load_models()
# player2.load_model_opponent(player1)

# Create the tensorboard writer
w = tensorboard.SummaryWriter()

# Training loop
N = configs["model"]["trajectory_length"]
n_games = configs["training"]["episodes"]

best_score = -1000
score_history = []
learn_iters = 0
n_steps = 0

# start timer
start = time.time()

for i in range(n_games):
    observation = env.reset()[0]
    observation2 = env.reset()[0]
    done = False
    score = 0
    game_length = 0
    while not done:
        action1, prob, val = player1.choose_action(observation)
        if configs["env"]["name"] == "tron": 
            action2, _, _ = player2.choose_action(observation2)
            observation_, reward, done, _ , info = env.step(action1, take_bot_action=action2)
        else:
            observation_, reward, done, _ , info = env.step(action1)  
        game_length += 1
        n_steps += 1
        score += reward
        
        player1.remember(observation, action1, prob, reward, done, val)
        if n_steps % N == 0:
            player1.learn()
            learn_iters += 1
        observation = observation_[0]
        observation2 = observation_[1]

        plt.imshow(observation)
        plt.show()
        
        
    score_history.append(score)
    delta_time = (time.time()-start)/60
    w.add_scalar("score", score, i)
    w.add_scalar("time", delta_time, i)
    avg_score = np.mean(score_history[-30:])
        
    if avg_score > best_score and learn_iters > 30:
        best_score = avg_score
        player1.save_models()
        player2.load_model_opponent(player1)
            
    print(f"Episode {i:<8} | score {score:<8.2f} | avg_score {avg_score:<8.2f} | time_steps {n_steps:<8} | learning_steps {learn_iters:<8} | time (min) {delta_time:<8.1f} | best average {best_score:.2f}")