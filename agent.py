
from model import PolicyNetwork, ValueNetwork
from memory import PPOMemory
import numpy as np
import torch 

class Agent:
    def __init__(self, env):
        self.actor  = PolicyNetwork(env.observation_space.shape)
        self.critic = ValueNetwork(env.observation_space.shape)
        self.memory = PPOMemory(batch_size=5)
    
        self.n_epochs = 4  # num. epochs per set of mini-batches
        self.gamma = 0.99  # discount
        self.lambd = 0.95  # regularization
        self.epsilon = 0.1 # clip
    
    def remember(self, state, action, prob, reward, done, val):
        self.memory.store_memory(state, action, prob, reward, done, val)
        
    def save_models(self):
        print("... Saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        
    def load_models(self):
        print("... Loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        
    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float32).to(self.actor.device)
        
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
    
    def learn(self):
        for _ in range(self.n_epochs):
            
            # Generate batches of transitions
            state_arr, action_arr, oldprob_arr, \
            val_arr, reward_arr, done_arr, batches = self.memory.generate_batches()

            # Calculate advantage
            T = len(reward_arr)
            advantage = np.zeros(T, dtype=np.float32)
            for t in range(T-1):
                a_t = 0
                for k in range(t, T-1):
                    delta = reward_arr[k] + \
                            self.gamma*val_arr[k+1]*(1-int(done_arr[k])) - \
                            val_arr[k]
                    a_t += (self.gamma*self.lambd)**k * delta
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)  
            values = torch.tensor(val_arr).to(self.actor.device)
            
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float32).to(self.actor.device)
                old_probs = torch.tensor(oldprob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp() 

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = advantage[batch] * torch.clamp(prob_ratio, 
                                                                        1-self.epsilon,
                                                                        1+self.epsilon)
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                
                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
        self.memory.clear_memory()