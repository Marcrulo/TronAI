from tron_env import TronEnv
from agent import Agent
import gym
import yaml

configs = yaml.safe_load(open("config.yaml"))

if configs["env"]["name"] == "tron":
    env = TronEnv(render_mode="human")
else:
    env = gym.make(configs["env"]["name"], render_mode="human")

player1 = Agent(num_actions=env.action_space.n, 
                num_observations=env.observation_space.shape)
player1.load_models()

# evaluate
for i in range(configs["eval"]["episodes"]):
    observation = env.reset()[0]
    done = False
    score = 0
    while not done:
        action, prob, val = player1.choose_action(observation)
        observation_, reward, done, _ , info = env.step(action)
        score += reward
        observation = observation_
    print(f"Episode {i}, score {score}")