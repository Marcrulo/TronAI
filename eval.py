
from agent import Agent
import gym

env = gym.make("CartPole-v1", render_mode="human")
player1 = Agent(env)
player1.load_models()

# evaluate
for i in range(100):
    observation = env.reset()[0]
    done = False
    score = 0
    while not done:
        action, prob, val = player1.choose_action(observation)
        observation_, reward, done, _ , info = env.step(action)
        score += reward
        observation = observation_
    print(f"Episode {i}, score {score}")