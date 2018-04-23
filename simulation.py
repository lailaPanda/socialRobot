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
import pandas as pd
import matplotlib.pyplot as plt
#required for path finding
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from operations import unwrapFeatures
from operations import getAngle
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor





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
NUM_AGENTS=10
AGENT_SIZE=15 #here width = height for simplicity

screendata = []
gridsize = 0


#load model and scalers from disk
reg = joblib.load('NNmodel.pkl') 
scaler_f = joblib.load('scaler_f.pkl') 
scaler_v = joblib.load('scaler_v.pkl') 

f=getFearureVector(agent)


#this returns the feature vector to be input to the motion prediction model
def getFearureVector(agent):
    [obs_mat,people_mat,people_velo,people_angle] = getAroundData(agent.rect.x,agent.rect.y)
    #get relative postion of the next goal of this agent
    goal = list(np.array([agent.rect.x,agent.rect.y]) - np.array(agent.path[agent.goalPos]))
    feature = [goal,obs_mat,people_mat,people_velo,people_angle,-1,-1] 
    [f,v] = unwrapFeatures(feature)
    return f
        
def predict(NNmodel,f):
    df = pd.DataFrame(data = [f])
    df_scaled = scaler_f.transform(df)
    pred = np.asmatrix(reg.predict(df_scaled))
    velo = pred[0,0]
    angle = getAngle(pred[0,1],pred[0,2])
    #rescale!!!!!

    
    
#get around data of position [x,y]
def getAroundData(x,y):
        #calculate the look around square in this dimention
        pixel_square_size = SQUARE_SIZE/SCALE
        pixel_grid_size = GRIDSIZE/SCALE
        obs_mat = np.zeros((int(pixel_square_size/pixel_grid_size),int(pixel_square_size/pixel_grid_size)))
        people_mat = np.zeros((int(pixel_square_size/pixel_grid_size),int(pixel_square_size/pixel_grid_size)))
        people_velo =  np.zeros((int(pixel_square_size/pixel_grid_size),int(pixel_square_size/pixel_grid_size)))
        people_angle =  np.zeros((int(pixel_square_size/pixel_grid_size),int(pixel_square_size/pixel_grid_size)))
    
        #this is the left,top coordinate of the look around square (coordinates in pygame start from left,top corner)
        [x0,y0] = [x-pixel_square_size/2,y-pixel_square_size/2]
        for i in range(0,obs_mat.shape[0]): #y coordinate
            for j in range(0,obs_mat.shape[1]): #x coordinate
                #create a rectangle of the grid
                gridRect = pygame.Rect((x0 + j * pixel_grid_size),(y0 + i * pixel_grid_size),(pixel_grid_size),(pixel_grid_size))
                [obs,plyr] = getMaxOvelappingThings(gridRect)
                maxObsOverlap = 0
                maxPlayerOvelap = 0
                if(obs.shape[0] > 0):
                    obs = obs.iloc[0]
                    maxObsOverlap = obs[1]/(gridRect.width * gridRect.height)
                if(plyr.shape[0] > 0):
                    plyr = plyr.iloc[0]
                    maxPlayerOvelap = plyr[1]/(plyr[0].rect.height * plyr[0].rect.width )
                #around_mat[i][j] = maxOverlap
                if(maxObsOverlap > 0.2):
                    obs_mat[i][j] = 1
                if(maxPlayerOvelap > 0.5):
                    people_mat[i][j]=1
                    people_velo[i][j] = math.sqrt(plyr[0].speed_x*plyr[0].speed_x + plyr[0].speed_y*plyr[0].speed_y)
                    people_angle[i][j] = math.atan2(plyr[0].speed_y,plyr[0].speed_x)
                    
        return [obs_mat,people_mat,people_velo,people_angle]

def getMap():
    gridsize = 50
    screendata = np.ones((int(SCREEN_HEIGHT/gridsize),int(SCREEN_WIDTH/gridsize)))
    
    for i in range(0,screendata.shape[0]):#y coordinate
        for j in range(0,screendata.shape[1]):#x coordinates
            gridRect = pygame.Rect((j*gridsize),(i*gridsize),(gridsize),(gridsize))
            [obs,plyr] = getMaxOvelappingThings(gridRect)
            maxObsOverlap = 0
            if(obs.shape[0] > 0):
                obs = obs.iloc[0]
                maxObsOverlap = obs[1]/(gridRect.width * gridRect.height)
            if(plyr.shape[0] > 0):
                plyr = plyr.iloc[0]
            if(maxObsOverlap > 0.2):
                screendata[i][j] = 0
    return [screendata,gridsize] 
    
def getPath(init_x,init_y,goal_x,goal_y):
    grid = Grid(matrix = screendata)
    start = grid.node(int(init_x/gridsize),int(init_y/gridsize))
    end = grid.node(int(goal_x/gridsize),int(goal_y/gridsize))
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)
    #print('operations:', runs, 'path length:', len(path))
    #print(grid.grid_str(path=path, start=start, end=end))
    path = np.array(path)*gridsize
    path[0] = [init_x,init_y]
    path[len(path)-1] = [goal_x,goal_y]
    #remove points in obstacles
    newpath = []
    for i in range(0,len(path)):
        [obs,plyr] = getMaxOvelappingThings(pygame.Rect((path[i][0]-AGENT_SIZE/2),(path[i][0] - AGENT_SIZE/2),(AGENT_SIZE),(AGENT_SIZE)))
        maxObsOverlap = 0
        maxPlayerOvelap = 0
        if(obs.shape[0] > 0):
            obs = obs.iloc[0]
            maxObsOverlap = obs[1]/(AGENT_SIZE*AGENT_SIZE)
        if(plyr.shape[0] > 0):
            plyr = plyr.iloc[0]
            maxPlayerOvelap = plyr[1]/(plyr[0].rect.height * plyr[0].rect.width ) 
        if(maxObsOverlap < 1 ):
            newpath.append([path[i][0],path[i][1]])
    return newpath

                           
            
def getMaxOvelappingThings(rect):
    sprites = all_sprite_list.sprites() 
    walls_overlap = []
    players_overlap = []
    for i in range(0,len(sprites)):
        if(type(sprites[i]) is Wall):
            walls_overlap.append([sprites[i],getOverlapArea(sprites[i].rect,rect)])
        if(type(sprites[i]) is Agent):
            players_overlap.append([sprites[i],getOverlapArea(sprites[i].rect,rect)])
               
    walls_overlap = pd.DataFrame(walls_overlap)
    players_overlap = pd.DataFrame(players_overlap)
    if(walls_overlap.shape[0] > 0):
        maximumWallOverlap = walls_overlap.loc[walls_overlap[1] == max(walls_overlap[1])]
    else:
        maximumWallOverlap = walls_overlap
    if(players_overlap.shape[0] > 0):
        maximumPlayerOverlap = players_overlap.loc[players_overlap[1] == max(players_overlap[1])]
    else:
        maximumPlayerOverlap = players_overlap        
    return [maximumWallOverlap,maximumPlayerOverlap]
            
        
    
         
def getOverlapArea(rect1,rect2):
       rect1X = [rect1.left,rect1.right]
       rect2X = [rect2.left,rect2.right]
       
       rect1Y = [rect1.top,rect1.bottom]
       rect2Y = [rect2.top,rect2.bottom]
       
       #get overlapping region
       overlapArea = getOverlapLineSeg(rect1X,rect2X) * getOverlapLineSeg(rect1Y,rect2Y)
       return overlapArea
       
def getOverlapLineSeg(p1,p2): # p1 = [x1,x2], p2 = [x3,x4]
    p1.sort()
    p2.sort()
    if((p1[1] < p2[0]) or (p2[1] < p1[0])):
        return 0
    #else there is an overlap
    if (p1[0] <= p2[0]):
        if(p1[1] <= p2[1]):
            overlap = p1[1] - p2[0]
        else:
            overlap = p2[1] - p2[0]
    else:
        if (p2[1] <= p1[1]):
            overlap = p2[1] - p1[0]
        else:
            overlap = p1[1] -p1[0]
    return overlap
    

class Agent(pygame.sprite.Sprite):
 
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
        self.speed_x = 0
        self.speed_y = 0
        self.walls = None
        self.path = []
        self.goalPos = 0
 
    def setSpeed(self, x, y):
        self.speed_x = x
        self.speed_y = y
    
    def setPath(self,path):
        self.path = path
    def update(self):
        """ Update the player position. """
        # Move left/right
        self.rect.x += self.speed_x
 
        # Did this update cause us to hit a wall?
        block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
        for block in block_hit_list:
            # If we are moving right, set our right side to the left side of
            # the item we hit
            if self.speed_x > 0:
                self.rect.right = block.rect.left
            else:
                # Otherwise if we are moving left, do the opposite.
                self.rect.left = block.rect.right
 
        # Move up/down
        self.rect.y += self.speed_y
 
        # Check and see if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, self.walls, False)
        for block in block_hit_list:
 
            # Reset our position based on the top/bottom of the object.
            if self.speed_y > 0:
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

def initAgent():
     itr = 0
     found = True
     #generate random rectangle for initial position of the agent
     initRect = pygame.Rect(random.randint(0,SCREEN_WIDTH),random.randint(0,SCREEN_HEIGHT),AGENT_SIZE,AGENT_SIZE)
     agent = -1
     while (isOccupied(initRect)):
         if(itr > 500):
             found = False
             break
         initRect = pygame.Rect(random.randint(0,SCREEN_WIDTH),random.randint(0,SCREEN_HEIGHT),AGENT_SIZE,AGENT_SIZE)
         itr+=1
     itr=0
     goalRect = pygame.Rect(random.randint(0, SCREEN_WIDTH),random.randint(0,SCREEN_HEIGHT),AGENT_SIZE,AGENT_SIZE)
     while((isOccupied(goalRect)) and found):
         if(itr > 500):
             found = False
             break
         goalRect = pygame.Rect(random.randint(0, SCREEN_WIDTH),random.randint(0,SCREEN_HEIGHT),AGENT_SIZE,AGENT_SIZE)
         itr +=1
     #get the path
     if(found):
         path = getPath((initRect.left + initRect.width/2),(initRect.top + initRect.height/2),(goalRect.left + goalRect.width/2),(goalRect.top + goalRect.height/2))
     if (len(path) == 0):
         found=False
     if(found):
         agent = Agent(initRect)
         agent.walls = wall_list
         #initial speed of this agent
         agent.setSpeed(0,0)
         agent.path = path
     return agent
         
    

def createPlayers():
    agent_rects = []
    for i in range(0,NUM_AGENTS):
        agent = initAgent()
        if (type(agent) is Agent):
            all_sprite_list.add(agent)
    return agent_rects
    
            
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
[screendata,gridsize]  = getMap() 
createPlayers()
 
clock = pygame.time.Clock()
 
done = False


while not done:
    
    #predict velocities and headings from the NN
    #update them
    #yey!
 
    all_sprite_list.update()
 
    screen.fill(BLACK)
 
    all_sprite_list.draw(screen)
 
    pygame.display.flip()
 
    clock.tick(60)
 
pygame.quit()



'''
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

'''
