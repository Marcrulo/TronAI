from tron_env import TronEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env


env = TronEnv(render_mode='human')

env.reset()
i = -1
while True:
    i += 1
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"Step: {i}")
    env.render('human')
    if done:
        print("Episode finished")
        break