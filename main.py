# from tron_env import TronEnv
from matplotlib import pyplot as plt
import os
from agent import Agent
import gym
import numpy as np
import pandas as pd

# env = TronEnv(render_mode='human')
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("LunarLander-v2")
N = 20

player1 = Agent(env)
n_games = 300

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
    avg_score = np.mean(score_history[-100:])
        
    if avg_score > best_score:
        best_score = avg_score
        player1.save_models()
            
    print(f"Episode {i}, score {score:.2f}, avg_score {avg_score:.2f},time_steps {n_steps},learning_steps {learn_iters}")

plt.plot(pd.Series(score_history).rolling(10).mean())
plt.show()