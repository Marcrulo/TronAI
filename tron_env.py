import random
import time
import pygame
import gymnasium as gym
from gymnasium import spaces # gym
import numpy as np
from player import Player
import yaml
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

configs = yaml.safe_load(open("config.yaml"))

WIDTH, HEIGHT = configs['game']['width'], configs['game']['height']
SCALE = configs['game']['scale']
ROW, COLUMN = WIDTH//SCALE, HEIGHT//SCALE
FPS = configs['game']['fps']

obs_size =  ROW*COLUMN + 2*ROW + 2*COLUMN

class TronEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=int)
        self.directions = ['right', 'left', 'up', 'down']

        if self.render_mode == 'human':
            pygame.init()
            self.display = pygame.display.set_mode((WIDTH+2*SCALE, HEIGHT+2*SCALE))
            pygame.display.set_caption('Tron')
            self.clock = pygame.time.Clock()

    def obs_to_img(self, obs, p1):
        img1 = obs.copy()
        img1[p1.y//SCALE, p1.x//SCALE] = 1
        return img1
    
    def step(self, action):
        
        self.observation = self.observation.reshape(ROW, COLUMN)

        if action == 0 and self.p1.dir != 'down':
            self.p1.dir = 'up'
        elif action == 1 and self.p1.dir != 'up':
            self.p1.dir = 'down'
        elif action == 2 and self.p1.dir != 'left':
            self.p1.dir = 'right'
        elif action == 3 and self.p1.dir != 'right':
            self.p1.dir = 'left'
        
        # self.observation = -np.ones((ROW, COLUMN))
        self.head = -np.ones(ROW+COLUMN+2)
        
        self.p1.update()
        self.head[self.p1.y//SCALE] = 1
        self.head[ROW+self.p1.x//SCALE] = 1
        for t in self.p1.trail:
            self.observation[t[1]//SCALE, t[0]//SCALE] = 0 #self.player_ids[i]
         
        if self.p1.alive:
            self.reward = 1
        elif not self.p1.alive:
            self.reward = -10000
            
        if self.p1.alive:
            self.done = False
        else:
            self.done = True

        self.observation = self.obs_to_img(self.observation, self.p1)
        
        if self.render_mode == "human":
            self.render()
        
        return self.observation, 1, self.done, {}, {}
            
    def reset(self, seed=None):
        self.seed = 0
        self.reward = 0
        
        # random start position and direction
        self.p1 = Player(start_pos=(np.random.randint(3,ROW-3)*SCALE, 
                                    np.random.randint(3,COLUMN-3)*SCALE),
                         start_dir=random.choice(self.directions),
                         scale=SCALE, width=WIDTH, height=HEIGHT)
        # self.p1 = Player(start_pos=((8*SCALE),(8*SCALE)),
        #                  start_dir=1, 
        #                  scale=SCALE, width=WIDTH, height=HEIGHT)
        
        self.observation = -np.ones((ROW, COLUMN))

        # add a padding to the observation image
        self.observation = np.array(ImageOps.expand(Image.fromarray(self.observation), border=1, fill=0))
        self.observation = self.obs_to_img(self.observation, self.p1)

        if self.render_mode == 'human':
            self.clock = pygame.time.Clock()
            self.render()
        
        return self.observation.flatten(), {}
        
    def render(self):
        # Background
        self.display.fill((67,70,75))
        
        # Borders
        pygame.draw.rect(self.display, 'WHITE', (SCALE, SCALE, WIDTH, 1))        # TOP
        pygame.draw.rect(self.display, 'WHITE', (SCALE, HEIGHT+SCALE, WIDTH, 1)) # BOTTOM 
        pygame.draw.rect(self.display, 'WHITE', (SCALE, SCALE, 1, HEIGHT))       # LEFT
        pygame.draw.rect(self.display, 'WHITE', (WIDTH+SCALE, SCALE, 1, HEIGHT)) # RIGHT
        
        # Players
        for t in self.p1.trail:
            pygame.draw.rect(self.display, (0, 150, 0), (t[0], t[1], SCALE, SCALE))
        pygame.draw.rect(self.display, (0, 255, 0), (self.p1.x, self.p1.y, SCALE, SCALE))
        
        pygame.display.update()
        self.clock.tick(FPS)
        
    def close(self):
        pygame.quit()