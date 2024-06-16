import random
import time
import pygame
import gym
from gym import spaces
import numpy as np
from player import Player

from matplotlib import pyplot as plt

WIDTH, HEIGHT = 620, 620
SCALE = 20
ROW, COLUMN = WIDTH//SCALE, HEIGHT//SCALE
FPS = 10

from model import PolicyNetwork
obs_size = ROW*COLUMN + 2*ROW + 2*COLUMN
Model = PolicyNetwork((obs_size,), 4)

class TronEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=int)
        self.directions = ['right', 'left', 'up', 'down']
        pygame.init()
        pygame.display.set_caption('Tron')
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        self.player_ids = [-1,1]

        
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
        
        
        self.observation = np.zeros((ROW, COLUMN))
        self.head1 = np.zeros(ROW+COLUMN)
        self.head2 = np.zeros(ROW+COLUMN)
        
        heads = [self.head1, self.head2]
        for i, player in enumerate(self.players):
            player.update(scale=SCALE, width=WIDTH, height=HEIGHT, p_opponent=self.players[not i])
            heads[i][player.y//SCALE] = 1
            heads[i][ROW+player.x//SCALE] = 1
            for t in player.trail:
                self.observation[t[1]//SCALE, t[0]//SCALE] = self.player_ids[i]
        self.observation = np.concatenate((self.observation.flatten(), self.head1, self.head2), axis=0)
                
        if self.p1.alive and not self.p2.alive:
            self.reward = 100
        elif not self.p1.alive and self.p2.alive:
            self.reward = -100
        else:
            self.reward = -1
            
        if self.p1.alive and self.p2.alive:
            self.done = False
        else:
            self.done = True

        if self.render_mode == 'human':
            self.render()
            
        return self.observation, self.reward, self.done, {}
            
            
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
        
        self.observation = np.zeros((ROW, COLUMN))
        self.head1 = np.zeros(ROW+COLUMN)
        self.head2 = np.zeros(ROW+COLUMN)
        
        self.head1[self.p1.y//SCALE] = 1
        self.head1[ROW+self.p1.x//SCALE] = 1
        self.head2[self.p2.y//SCALE] = 1
        self.head2[ROW+self.p2.x//SCALE] = 1
        
        self.observation = np.concatenate((self.observation.flatten(), self.head1, self.head2), axis=0)
        
        if self.render_mode == 'human':
            self.clock = pygame.time.Clock()
            self.render()
            
        return self.observation
        
    def render(self, render_mode='human'):
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

        # render env observation
        # print(self.observation[ROW*COLUMN:-ROW-COLUMN])
        # print(self.observation[ROW*COLUMN+ROW+COLUMN:])
        # plt.imshow(self.observation[:ROW*COLUMN].reshape(ROW,COLUMN), cmap='viridis', interpolation='nearest')
        # plt.show()
        
        

        pygame.display.update()
        self.clock.tick(FPS)
        
    def close(self):
        pygame.quit()