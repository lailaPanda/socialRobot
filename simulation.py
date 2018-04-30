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
from pathfinding.finder.breadth_first import BreadthFirstFinder

from operations import unwrapFeatures
from operations import getAngle
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from operations import getAngleFromCoords






#these should ne imported from the other file. Do this later
SQUARE_SIZE = 5000 #the size of the square the person looks around this is in mm
GRIDSIZE = 1000  #grids would be GRIDSIZE * GIRDSIZE . this is in mm
 
# -- Global constants

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 50, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
 

#real world dimentions of the game world in mm
WIDTH = 200000
HEIGHT = 200000
#scale-down to the screen
SCALE = 200
# Screen dimensions
SCREEN_WIDTH = math.floor(WIDTH/SCALE)
SCREEN_HEIGHT = math.floor(HEIGHT/SCALE)
 
#agents
NUM_AGENTS=10
AGENT_SIZE=15 #here width = height for simplicity
MAX_VELO = 5000 # this is in mm

#load model and scalers from disk
reg = joblib.load('NNmodel.pkl') 
scaler_f = joblib.load('scaler_f.pkl') 
scaler_v = joblib.load('scaler_v.pkl') 

f = [[-0.5,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
reg.predict(f)


#this returns the feature vector to be input to the motion prediction model
def getFearureVector(agent):
    [around_mat,people_velo_x,people_velo_y] = getAroundData(agent.rect.x,agent.rect.y)
    #get relative postion of the next goal of this agent
    goal_pos = list(np.array(agent.path[agent.goalPos]) - np.array([agent.rect.x,agent.rect.y]))
    #normalize goal pos vector
    goal_sum = abs(goal_pos[0]) + abs(goal_pos[1])
    goal_pos[0] /= goal_sum
    goal_pos[1] /= goal_sum
    print("agent position = " + str([agent.rect.x,agent.rect.y]))
    print("goal position = " + str(agent.path[agent.goalPos]))
    print("goal_pos = " + str(goal_pos))
    feature = [goal_pos[0],goal_pos[1],around_mat,people_velo_x,people_velo_y,-1,-1] 
    [f,v] = unwrapFeatures(feature)
    return f
        
def predict(f):
    df = pd.DataFrame(data = [f])
    pred = reg.predict(df)[0]
    #simple rescaling pred
    pred_sum = abs(pred[0]) + abs(pred[1])
    pred[0] /= pred_sum
    pred[1] /= pred_sum
    
    '''
    #pred = scaler_v.inverse_transform(pred)
    velo = pred[0,0]
    sin = pred[0,1]
    cos = pred[0,2]
    #scale back velo to fit screen. The predicted velo is in mm/s
    #velo = velo/(SCALE)
    
    if(pred[0,1] > 1): pred[0,1] = 1
    if(pred[0,1] < -1): pred[0,1] = -1
    if(pred[0,2] > 1): pred[0,2] = 1
    if(pred[0,2] < -1): pred[0,2] = -1
    angle = getAngle(pred[0,1],pred[0,2])
    '''
    return pred

def nextVelo_allAgents():
    #iterate through all agents
    for i in range(0,len(all_sprite_list.sprites())):
        if(type(all_sprite_list.sprites()[i]) is Agent):
            f = getFearureVector(all_sprite_list.sprites()[i])
            pred = predict(f)
            print("pred = " + str(pred))
            '''
            if(f[0]*sin < 0): sin = (sin + f[0])/2
            if(f[1]*cos < 0): cos = (cos + f[1])/2
            if(velo < 0): velo = 0
            if(velo > 1): velo=1
            velo*=100
            '''
            all_sprite_list.sprites()[i].setSpeed(pred[0]*8,pred[1]*8)
            '''
            print("sin, cos = " + str([sin,cos]))
            print("abs velo : " + str(velo))
            print("velocity : " + str(velo*cos) + " " + str(velo*sin))
            print("angle : " + str(getAngle(sin,cos)*180/math.pi))
            '''
            
            

#for testing
def printAgentData():
    for i in range(0,len(all_sprite_list.sprites())):
        agent = all_sprite_list.sprites()[i]
        if(type(agent) is Agent):
            print("agent" + str(i))
            print([agent.rect.x,agent.rect.y,agent.goalPos,agent.path,agent.speed_x,agent.speed_y])
            print("dist to tmp goal = " + str(abs(agent.rect.x - agent.path[agent.goalPos][0]) + abs(agent.rect.y - agent.path[agent.goalPos][1])))
    
    
#get around data of position [x,y]
def getAroundData(x,y):
        #calculate the look around square in this dimention
        pixel_square_size = SQUARE_SIZE/SCALE
        pixel_grid_size = GRIDSIZE/SCALE
        obs_mat = np.zeros((int(pixel_square_size/pixel_grid_size),int(pixel_square_size/pixel_grid_size)))
        people_mat = np.zeros((int(pixel_square_size/pixel_grid_size),int(pixel_square_size/pixel_grid_size)))
        people_velo_x =  np.zeros((int(pixel_square_size/pixel_grid_size),int(pixel_square_size/pixel_grid_size)))
        people_velo_y =  np.zeros((int(pixel_square_size/pixel_grid_size),int(pixel_square_size/pixel_grid_size)))
    
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
                    people_velo_x[i][j] = math.sqrt(plyr[0].speed_x)
                    people_velo_y[i][j] = math.sqrt(plyr[0].speed_y)
                
        around_mat =  (np.logical_or(people_mat,obs_mat)).astype(int)            
        return [around_mat,people_velo_x,people_velo_y]

def getMap():
    screendata = np.ones((int(SCREEN_HEIGHT/mapgridsize),int(SCREEN_WIDTH/mapgridsize)))
    
    for i in range(0,screendata.shape[0]):#y coordinate
        for j in range(0,screendata.shape[1]):#x coordinates
            gridRect = pygame.Rect((j*mapgridsize),(i*mapgridsize),(mapgridsize),(mapgridsize))
            [obs,plyr] = getMaxOvelappingThings(gridRect)
            maxObsOverlap = 0
            if(obs.shape[0] > 0):
                obs = obs.iloc[0]
                maxObsOverlap = obs[1]/(gridRect.width * gridRect.height)
            if(plyr.shape[0] > 0):
                plyr = plyr.iloc[0]
            if(maxObsOverlap > 0.2):
                screendata[i][j] = 0
    return screendata
    
def getPath(init_x,init_y,goal_x,goal_y):
    grid = Grid(matrix = screendata)
    start = grid.node(int(init_x/mapgridsize),int(init_y/mapgridsize))
    end = grid.node(int(goal_x/mapgridsize),int(goal_y/mapgridsize))
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    finder = BreadthFirstFinder(diagonal_movement=DiagonalMovement.always)
        
    path, runs = finder.find_path(start, end, grid)
    #print('operations:', runs, 'path length:', len(path))
    print(grid.grid_str(path=path, start=start, end=end))
    path = np.array(path)*mapgridsize
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
    
class Point(pygame.sprite.Sprite):
    # Constructor function
    def __init__(self,rect,col):
        # Call the parent's constructor
        super().__init__()
        [x,y] = [rect.left,rect.top]
        # Set height, width
        self.image = pygame.Surface([rect.width, rect.width])
        self.image.fill(col)
 
        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x
 
        
class Agent(pygame.sprite.Sprite):
 
    # Constructor function
    def __init__(self,rect,COL):
        # Call the parent's constructor
        super().__init__()
        [x,y] = [rect.left,rect.top]
        # Set height, width
        self.image = pygame.Surface([rect.width, rect.width])
        self.image.fill(COL)
 
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
        self.goalPoints = []
        self.reached_goal = False
 
    def setSpeed(self, x, y):
        self.speed_x = x
        self.speed_y = y
    
    def setPath(self,path):
        self.path = path
    def update(self):
        print("in update")
        #update current goal possition
        dist_to_goal = (abs(self.rect.x - self.path[self.goalPos][0]) + abs(self.rect.y - self.path[self.goalPos][1]))
        if(dist_to_goal < 20):
            self.goalPos+=1
            if(self.goalPos >= len(self.path)):
                self.goalPos = (len(self.path) - 1)
                self.reached_goal = True
                self.speed_x = 0
                self.speed_y = 0
                print("here!")
        #color current goal
        for i in range(0,len(all_sprite_list.sprites())):
            if(type(all_sprite_list.sprites()[i]) is Point):
                point = all_sprite_list.sprites()[i]
                dist = abs(self.path[self.goalPos][0] - point.rect.x) + abs(self.path[self.goalPos][1] - point.rect.y)
                if (dist < 1):
                    all_sprite_list.sprites()[i].image.fill((0,0,0))
                else:
                    all_sprite_list.sprites()[i].image.fill((0,0,0))
                    
                #print("dist to goal : " + str(dist))
                #print("goal pos = " + str(self.goalPos))
            
        """ Update the player position. """
        # Move left/right
        self.rect.x += int(self.speed_x)#times step
 
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
        self.rect.y += int(self.speed_y) #timestep
 
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
     col = (0,0,0)
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
         col = ((random.randint(30, 255)),(random.randint(30, 255)),(random.randint(30, 255)))
         agent = Agent(initRect,col)
         agent.walls = wall_list
         #initial speed of this agent
         agent.setSpeed(0,0)
         agent.path = path
         #sometimes the goal is not in the path! due to scaling and obstacle avoidance. so add it manually
         goalRect = pygame.Rect((path[len(path)-1][0]),(path[len(path)-1][1]),(AGENT_SIZE/4),(AGENT_SIZE/4))
     return [agent,goalRect,col]
         
  

def createPlayers(plotGoals,plotPaths):
    for i in range(0,NUM_AGENTS):
        [agent,goalRect,col] = initAgent()
        if (type(agent) is Agent):
            print("adding agent")
            all_sprite_list.add(agent)
            if(plotGoals):
                all_sprite_list.add(Point(goalRect,col)
            if(plotPaths):
                for j in range(0,len(agent.path)):
                    rect = pygame.Rect(agent.path[j][0],agent.path[j][1],5,5)
                    all_sprite_list.add(Point(rect,col))
    
    
          
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
mapgridsize = 30
screendata = getMap() 
createPlayers(True,True)
 
clock = pygame.time.Clock()
 
done = False


while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    #predict velocities and headings from the NN
    #update them
    #yey!
    print("runnning..")
    all_sprite_list.update()
    nextVelo_allAgents()
 
    screen.fill(BLACK)
 
    all_sprite_list.draw(screen)
 
    pygame.display.flip()
    #printAgentData()
 
    clock.tick(10)
 
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
