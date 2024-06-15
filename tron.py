import pygame
import random
import numpy as np
from player import Player



WIDTH, HEIGHT = 620, 620
SCALE = 20
ROW, COLUMN = WIDTH//SCALE, HEIGHT//SCALE
FPS = 10




# Initialize
pygame.init()
pygame.display.set_caption('Tron')
display = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

p1 = Player(start_pos=(20,20),
            start_dir='right')
p2 = Player(start_pos=(500,500),
            start_dir='left')

# Game loop
while running:
    
    # Events
    for event in pygame.event.get():
        # Quit button
        if event.type == pygame.QUIT:
            running = False
            break
            
        # Keypress
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break
            
            if event.key == pygame.K_d and p1.dir != 'left':
                p1.dir = 'right'
            if event.key == pygame.K_a and p1.dir != 'right':
                p1.dir = 'left'
            if event.key == pygame.K_s and p1.dir != 'up':
                p1.dir = 'down'
            if event.key == pygame.K_w and p1.dir != 'down':
                p1.dir = 'up'
                
            if event.key == pygame.K_RIGHT and p2.dir != 'left':
                p2.dir = 'right'
            if event.key == pygame.K_LEFT and p2.dir != 'right':
                p2.dir = 'left'
            if event.key == pygame.K_DOWN and p2.dir != 'up':
                p2.dir = 'down'
            if event.key == pygame.K_UP and p2.dir != 'down':
                p2.dir = 'up'
                
    # Background
    display.fill((67,70,75))
    
    # Borders
    # pygame.draw.rect(DISPLAY, COLOR, (LEFT, TOP, WIDTH, HEIGHT))
    pygame.draw.rect(display, 'WHITE', (SCALE, SCALE, WIDTH-2*SCALE, 1))        # TOP
    pygame.draw.rect(display, 'WHITE', (SCALE, HEIGHT-SCALE, WIDTH-2*SCALE, 1)) # BOTTOM 
    pygame.draw.rect(display, 'WHITE', (SCALE, SCALE, 1, HEIGHT-2*SCALE))       # LEFT
    pygame.draw.rect(display, 'WHITE', (WIDTH-SCALE, SCALE, 1, HEIGHT-2*SCALE)) # RIGHT
    
    # Players
    players = [p1, p2]
    for i, player in enumerate(players):
        player.update(scale=SCALE, width=WIDTH, height=HEIGHT, p_opponent=players[not i])
        for t in player.trail:
            pygame.draw.rect(display, (0, (not i)*150,i*150), (t[0]+SCALE, t[1]+SCALE, SCALE, SCALE))
        pygame.draw.rect(display, (0, (not i)*255,i*255), (player.x+SCALE, player.y+SCALE, SCALE, SCALE))

    
    
    # Finalize update
    pygame.display.update()
    clock.tick(FPS)
    
pygame.quit()
        
             
