from tron_env import TronEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env


env = TronEnv(render_mode=None)
# check_env(env, warn=True)
# env = make_vec_env(lambda: GymV26Compatibility(TronEnv()), n_envs=1)

# env = gym.make("LunarLander-v2", render_mode="human")
# env = make_vec_env("LunarLander-v3", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1e+4)
model.save("tron")
# model = PPO.load("tron")


obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")

env.reset()
i = -1
while True:
    i += 1
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"Step: {i}")
    if env.render_mode == 'human':
        env.render()
    if done:
        print("Episode finished")
        break