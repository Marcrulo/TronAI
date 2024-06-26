import numpy as np
import torch

class PPOMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.clear_memory()
        
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []
        self.vals = []
        
    def store_memory(self, state, action, prob, reward, done, val):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.vals.append(val)
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states),\
               np.array(self.actions),\
               np.array(self.probs),\
               np.array(self.vals),\
               np.array(self.rewards),\
               np.array(self.dones),\
               batches