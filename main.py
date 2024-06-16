from tron_env import TronEnv
from matplotlib import pyplot as plt
import os
from agent import Agent
import gym
import numpy as np
import pandas as pd
from torch.utils import tensorboard
import yaml

configs = yaml.safe_load(open("config.yaml"))

if configs["env"]["tron"]:
    env = TronEnv(render_mode=configs["env"]["render_mode"])
else:
    env = gym.make(configs["env"]["name"], render_mode=configs["env"]["render_mode"])

player1 = Agent(num_actions=env.action_space.n, 
                num_observations=env.observation_space.shape)
w = tensorboard.SummaryWriter()
N = configs["model"]["trajectory_length"]
n_games = configs["training"]["episodes"]

best_score = -1000
score_history = []
learn_iters = 0
n_steps = 0

for i in range(n_games):
    observation = env.reset()[0]
    done = False
    score = 0
    while not done:
        action, prob, val = player1.choose_action(observation)
        observation_, reward, done, _ , info = env.step(action)
        n_steps += 1
        score += reward
        player1.remember(observation, action, prob, reward, done, val)
        if n_steps % N == 0:
            player1.learn()
            learn_iters += 1
        observation = observation_
    score_history.append(score)
    w.add_scalar("score", score, i)
    avg_score = np.mean(score_history[-100:])
        
    if avg_score > best_score:
        best_score = avg_score
        player1.save_models()
            
    print(f"Episode {i}, score {score:.2f}, avg_score {avg_score:.2f},time_steps {n_steps},learning_steps {learn_iters}")

plt.plot(pd.Series(score_history).rolling(10).mean())
plt.show()