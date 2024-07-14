import random
import time
import pygame
import gym
from gym import spaces
import numpy as np
from player import Player
import yaml
from matplotlib import pyplot as plt

configs = yaml.safe_load(open("config.yaml"))

WIDTH, HEIGHT = configs['game']['width'], configs['game']['height']
SCALE = configs['game']['scale']
ROW, COLUMN = WIDTH//SCALE, HEIGHT//SCALE
FPS = configs['game']['fps']

cnn = configs["env"]["cnn"]

from model import PolicyNetwork
obs_size = ROW*COLUMN + 2*ROW + 2*COLUMN
Model = PolicyNetwork((obs_size,), 4, cnn=True)

class TronEnv(gym.Env):
    def __init__(self, render_mode=None, cnn=False):
        self.render_mode = render_mode
        self.cnn = cnn
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=int)
        self.directions = ['right', 'left', 'up', 'down']
        self.player_ids = [2,-2]

        if self.render_mode == 'human':
            pygame.init()
            self.display = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption('Tron')
            self.clock = pygame.time.Clock()

    def obs_to_img(self, obs, p1, p2):
        img1 = obs.copy()
        img1[p1.y//SCALE, p1.x//SCALE] = self.player_ids[0]
        img1[p2.y//SCALE, p2.x//SCALE] = self.player_ids[1]
        # img = np.expand_dims(img, axis=0)

        img2 = obs.copy()
        img2[p1.y//SCALE, p1.x//SCALE] = self.player_ids[1]
        img2[p2.y//SCALE, p2.x//SCALE] = self.player_ids[0]

        return img1, img2
        
    def step(self, action, take_bot_action = False):
                
        if action == 0 and self.p1.dir != 'down':
            self.p1.dir = 'up'
        elif action == 1 and self.p1.dir != 'up':
            self.p1.dir = 'down'
        elif action == 2 and self.p1.dir != 'left':
            self.p1.dir = 'right'
        elif action == 3 and self.p1.dir != 'right':
            self.p1.dir = 'left'


        if take_bot_action:
            bot_action = take_bot_action
        else:
            bot_action = random.randint(0,3)
        

        if bot_action == 0 and self.p2.dir != 'down':
            self.p2.dir = 'up'
        elif bot_action == 1 and self.p2.dir != 'up':
            self.p2.dir = 'down'
        elif bot_action == 2 and self.p2.dir != 'left':
            self.p2.dir = 'right'
        elif bot_action == 3 and self.p2.dir != 'right':
            self.p2.dir = 'left'
        
        
        self.observation = -np.ones((ROW, COLUMN))
        self.head1 = -np.ones(ROW+COLUMN)
        self.head2 = -np.ones(ROW+COLUMN)
        
        heads = [self.head1, self.head2]
        for i, player in enumerate(self.players):
            player.update(scale=SCALE, width=WIDTH, height=HEIGHT, p_opponent=self.players[not i])
            heads[i][player.y//SCALE] = 1
            heads[i][ROW+player.x//SCALE] = 1
            for t in player.trail:
                self.observation[t[1]//SCALE, t[0]//SCALE] = 0 #self.player_ids[i]
                
        if self.p1.alive and not self.p2.alive:
            self.reward = 100
        elif not self.p1.alive:
            self.reward = -100
        else:
            self.reward = -1
            
        if self.p1.alive and self.p2.alive:
            self.done = False
        else:
            self.done = True

        if self.cnn:
            self.observation, self.observation2 = self.obs_to_img(self.observation, self.p1, self.p2)
        else:
            self.observation = np.concatenate((self.observation.flatten(), self.head1, self.head2), axis=0)
            self.observation2 = None
        
        if self.render_mode == "human":
            self.render()
        
            
        # return self.observation, self.reward, self.done, {}, {}
        return (self.observation, self.observation2), 1, self.done, {}, {}
            
    def reset(self):
        self.reward = 0
        
        # random start position and direction
        self.p1 = Player(start_pos=(np.random.randint(2,ROW-2)*SCALE, 
                                    np.random.randint(2,COLUMN-2)*SCALE),
                         start_dir=random.choice(self.directions))
        self.p2 = Player(start_pos=(np.random.randint(2,ROW-2)*SCALE, 
                                    np.random.randint(2,COLUMN-2)*SCALE),
                         start_dir=random.choice(self.directions))
        self.players = [self.p1, self.p2]
        
        self.observation = -np.ones((ROW, COLUMN))
        self.head1 = -np.ones(ROW+COLUMN)
        self.head2 = -np.ones(ROW+COLUMN)
        
        self.head1[self.p1.y//SCALE] = 1
        self.head1[ROW+self.p1.x//SCALE] = 1
        self.head2[self.p2.y//SCALE] = 1
        self.head2[ROW+self.p2.x//SCALE] = 1
        
        
        if self.cnn:
            self.observation, _ = self.obs_to_img(self.observation, self.p1, self.p2)
        else:
            self.observation = np.concatenate((self.observation.flatten(), self.head1, self.head2), axis=0)

        if self.render_mode == 'human':
            self.clock = pygame.time.Clock()
            self.render()
        
            
        return [self.observation,{}]
        
    def render(self):
        # Background
        self.display.fill((67,70,75))
        
        # Borders
        pygame.draw.rect(self.display, 'WHITE', (SCALE, SCALE, WIDTH-2*SCALE, 1))        # TOP
        pygame.draw.rect(self.display, 'WHITE', (SCALE, HEIGHT-SCALE, WIDTH-2*SCALE, 1)) # BOTTOM 
        pygame.draw.rect(self.display, 'WHITE', (SCALE, SCALE, 1, HEIGHT-2*SCALE))       # LEFT
        pygame.draw.rect(self.display, 'WHITE', (WIDTH-SCALE, SCALE, 1, HEIGHT-2*SCALE)) # RIGHT
        
        # Players
        for i, player in enumerate(self.players):
            for t in player.trail:
                pygame.draw.rect(self.display, (0, (not i)*150,i*150), (t[0]+SCALE, t[1]+SCALE, SCALE, SCALE))
            pygame.draw.rect(self.display, (0, (not i)*255,i*255), (player.x+SCALE, player.y+SCALE, SCALE, SCALE))
        
        pygame.display.update()
        self.clock.tick(FPS)
        
    def close(self):
        pygame.quit()