#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 17:00:42 2018

@author: sleek_eagle
"""

"""
Sample Python/Pygame Programs
Simpson College Computer Science
http://programarcadegames.com/
http://simpson.edu/computer-science/
 
From:
http://programarcadegames.com/python_examples/f.php?file=move_with_walls_example
 
Explanation video: http://youtu.be/8IRyt7ft7zg
 
Part of a series:
http://programarcadegames.com/python_examples/f.php?file=move_with_walls_example.py
http://programarcadegames.com/python_examples/f.php?file=maze_runner.py
http://programarcadegames.com/python_examples/f.php?file=platform_jumper.py
http://programarcadegames.com/python_examples/f.php?file=platform_scroller.py
http://programarcadegames.com/python_examples/f.php?file=platform_moving.py
http://programarcadegames.com/python_examples/sprite_sheets/
"""
 
import pygame
import math
import random
import numpy as np



#these should ne imported from the other file. Do this later
SQUARE_SIZE = 11000 #the size of the square the person looks around this is in mm
GRIDSIZE = 1000  #grids would be GRIDSIZE * GIRDSIZE . this is in mm
 
# -- Global constants

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 50, 255)
 

#real world dimentions of the game world in mm
WIDTH = 20000
HEIGHT = 20000
#scale-down to the screen
SCALE = 20
# Screen dimensions
SCREEN_WIDTH = math.floor(WIDTH/SCALE)
SCREEN_HEIGHT = math.floor(HEIGHT/SCALE)
 
#players
NUM_PLAYERS=5
PLAYER_SIZE=15 #here width = height for simplicity

#get around data of position [x,y]
def getAroundData(x,y):
    #calculate the look around square in this dimention
    pixel_square_size = SQUARE_SIZE/SCALE
    around_mat = np.zeros((int(pixel_square_size*2),int(pixel_square_size*2)))
    #this is the left,top coordinate of the look around square (coordinates in pygame start from left,top corner)
    [x0,y0] = [x-pixel_square_size/2,y-pixel_square_size/2]
    for i in range(0,around_mat.shape[0]):
        for j in range(0,around_mat.shape[1]):
            #create a rectangle
            pygame.Rect((x0 + i*),(),(),())
            
            
            
    
    
 
class Player(pygame.sprite.Sprite):
    """ This class represents the bar at the bottom that the player
    controls. """
 
    # Constructor function
    def __init__(self,rect):
        # Call the parent's constructor
        super().__init__()
        [x,y] = [rect.left,rect.top]
        # Set height, width
        self.image = pygame.Surface([rect.width, rect.width])
        self.image.fill(WHITE)
 
        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x
 
        # Set speed vector
        self.change_x = 0
        self.change_y = 0
        self.walls = None
 
    def changespeed(self, x, y):
        """ Change the speed of the player. """
        self.change_x += x
        self.change_y += y
 
    def update(self):
        """ Update the player position. """
        # Move left/right
        self.rect.x += self.change_x
 
        # Did this update cause us to hit a wall?
        block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
        for block in block_hit_list:
            # If we are moving right, set our right side to the left side of
            # the item we hit
            if self.change_x > 0:
                self.rect.right = block.rect.left
            else:
                # Otherwise if we are moving left, do the opposite.
                self.rect.left = block.rect.right
 
        # Move up/down
        self.rect.y += self.change_y
 
        # Check and see if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
        for block in block_hit_list:
 
            # Reset our position based on the top/bottom of the object.
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            else:
                self.rect.top = block.rect.bottom
 
 
class Wall(pygame.sprite.Sprite):
    """ Wall the player can run into. """
    def __init__(self, x, y, width, height):
        """ Constructor for the wall that the player can run into. """
        # Call the parent's constructor
        super().__init__()
 
        # Make a blue wall, of the size specified in the parameters
        self.image = pygame.Surface([width, height])
        self.image.fill(BLUE)
 
        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x
        

MAP_PATH = "/home/sleek_eagle/research/human_simulation/project/map.txt"
#read the low res map from file
def readMap():
    mapList = []
    file = open(MAP_PATH, 'r') 
    for line in file:
        if (len(line) > 1):
            chars = list(line)
            nums = []
            for i in range(0,len(chars)):
                if (chars[i].isdigit()):
                    nums.append(int(chars[i]))        
            mapList.append(nums)
    return mapList

#expand the low res map to fit into the screen
def createWalls(mapList):
    square_walls = []
    #calculate scale in y axis
    yscale = SCREEN_HEIGHT/len(mapList)
    if(len(mapList) > 0):
        xscale = SCREEN_WIDTH/len(mapList[0])
    else:
        print("data is not consistant!")
        return
    
    for i in range(len(mapList)):
        for j in range(len(mapList[i])):
            if (mapList[i][j] == 1):
                wall = Wall((j*xscale),(i*yscale), xscale, yscale)
                square_walls.append(wall)
    return square_walls

def isOccupied(rect):   
    #now check other players and walls 
    sprites = all_sprite_list.sprites() 
    for i in range(0,len(sprites)):
        if(rect.colliderect(sprites[i].rect)):
            return True
    return False

def createPlayerRects():
    player_rects = []
    for i in range(0,NUM_PLAYERS):
        #generate random rectangle
        rect = pygame.Rect(random.randint(0,SCREEN_WIDTH),random.randint(0,SCREEN_HEIGHT),PLAYER_SIZE,PLAYER_SIZE)
        while (isOccupied(rect)):
            rect = pygame.Rect(random.randint(0,SCREEN_WIDTH),random.randint(0,SCREEN_HEIGHT),PLAYER_SIZE,PLAYER_SIZE)
        player_rects.append(rect)
        
        player = Player(rect)
        player.walls = wall_list
        all_sprite_list.add(player)
    return player_rects


    
        

            
    

                
                
            
    
    
    
            
# Call this function so the Pygame library can initialize itself
pygame.init()    
# Create an 800x600 sized screen
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])    
# Set the title of the window
pygame.display.set_caption('Skynet')
 
# List to hold all the sprites
all_sprite_list = pygame.sprite.Group()
 
# Make the walls. (x_pos, y_pos, width, height)
wall_list = pygame.sprite.Group()

#read low res map from file
mapList = readMap()
#create scales version of walls from the low res map data
wallList = createWalls(mapList)

for i in range(0,len(wallList)):
    wall_list.add(wallList[i])
    all_sprite_list.add(wallList[i])
 
# Create the player paddle object
#choose a place for a player 

#player_rects = createPlayerRects()
#player = Player(player_rects[0])
#player.walls = wall_list
#all_sprite_list.add(player)
    
createPlayerRects()


 
clock = pygame.time.Clock()
 
done = False
 
while not done:
 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
 
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                player.changespeed(-3, 0)
            elif event.key == pygame.K_RIGHT:
                player.changespeed(3, 0)
            elif event.key == pygame.K_UP:
                player.changespeed(0, -3)
            elif event.key == pygame.K_DOWN:
                player.changespeed(0, 3)
 
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                player.changespeed(3, 0)
            elif event.key == pygame.K_RIGHT:
                player.changespeed(-3, 0)
            elif event.key == pygame.K_UP:
                player.changespeed(0, 3)
            elif event.key == pygame.K_DOWN:
                player.changespeed(0, -3)
 
    all_sprite_list.update()
 
    screen.fill(BLACK)
 
    all_sprite_list.draw(screen)
 
    pygame.display.flip()
 
    clock.tick(60)
 
pygame.quit()
