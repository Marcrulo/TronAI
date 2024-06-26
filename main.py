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
if configs["env"]["render_mode"] == "rgb_array":
    cnn = True

if configs["env"]["name"] == "tron":
    env = TronEnv(render_mode=configs["env"]["render_mode"])
else:
    env = gym.make(configs["env"]["name"], render_mode=configs["env"]["render_mode"])
_ = env.reset()

if cnn:
    obs_space = env.render().shape
else:
    obs_space = env.observation_space.shape
    
    
player1 = Agent(num_actions=env.action_space.n, 
                num_observations=obs_space,
                cnn=cnn)
player2 = Agent(num_actions=env.action_space.n,
                num_observations=obs_space,
                cnn=cnn)


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
    game_length = 0
    while not done:
        if cnn:
            observation = env.render()
        action1, prob, val = player1.choose_action(observation)
        if configs["env"]["name"] == "tron": 
            action2, _, _ = player2.choose_action(observation)
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
        observation = observation_
    score_history.append(score)
    w.add_scalar("score", score, i)
    w.add_scalar("game length", game_length, i)
    avg_score = np.mean(score_history[-100:])
        
    if avg_score > best_score:
        best_score = avg_score
        player1.save_models()
            
    print(f"Episode {i:<8} score {score:<8.2f} avg_score {avg_score:<8.2f} time_steps {n_steps:<8} learning_steps {learn_iters:<8} best average {best_score}")

plt.plot(pd.Series(score_history).rolling(10).mean())
plt.show()