from tron_env import TronEnv
from agent import Agent
import gym
import yaml
from matplotlib import pyplot as plt

configs = yaml.safe_load(open("config.yaml"))

if configs["env"]["name"] == "tron":
    env = TronEnv(render_mode="human")
else:
    env = gym.make(configs["env"]["name"], render_mode="human")
_ = env.reset()




obs_space = env.observation.shape
obs_space = (obs_space[0]-2, obs_space[1]-2)

player1 = Agent(num_actions=env.action_space.n, 
                num_observations=obs_space)
player2 = Agent(num_actions=env.action_space.n,
                num_observations=obs_space)
player1.load_models()
player2.load_model_opponent(player1)

# evaluate
for i in range(configs["eval"]["episodes"]):
    observation = env.reset()[0]
    observation2 = env.reset()[0]
    done = False
    score = 0
    while not done:
        action1, prob, val = player1.choose_action(observation)
        if configs["env"]["name"] == "tron": 
            action2, _, _ = player2.choose_action(observation2)
            observation_, reward, done, _ , info = env.step(action1, take_bot_action=action2)
        else:
            observation_, reward, done, _ , info = env.step(action1) 

        observation = observation_[0]
        observation2 = observation_[1]
        
        score += reward
    print(f"Episode {i}, score {score}")