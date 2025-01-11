from tron_env import TronEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env


env = TronEnv()
# check_env(env, warn=True)

# env = gym.make("LunarLander-v2", render_mode="human")
# env = make_vec_env("LunarLander-v3", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1e+5)
model.save("tron")
# model = PPO.load("tron") 


vec_env = model.get_env()
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, _, _ = env.step(action)
    env.render("human")