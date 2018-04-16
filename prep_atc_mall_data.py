#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 13:39:55 2018

@author: sleek_eagle
"""

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from pylab import *




data = pd.read_csv("/home/sleek_eagle/research/human_simulation/data/atc-20121114.csv",sep = ",",header=None,names =['ts','pid','posX','posY','posZ','velo','moAngle','faceAngle'])
cords = data[['posX','posY']]
[mapMat,GRIDSIZE] = createMap(cords)

#number of uniqie people in the dataset
upids = data.pid.unique()
len(data.pid.unique())
getAroundPeople(upids,1)


#if you get SettingWithCopyWarning refer https://maxpowerwastaken.github.io/blog/pandas_view_vs_copy/

               
def getAroundPeople(upids,personLimit):
    SQUARE_SIZE = 5000 #the size of the square the person looks around
    training = []
    itr = 0
    for pid in upids:
        itr+=1
        if (itr > personLimit):
            break
        thisPerson = (data['pid'] == pid)
        thisPerson = data[thisPerson]
        #entries for this person
        len(thisPerson.index)
        #get closest n number of people for this entry of this person
        i=0
        for index, thisPersonInst in thisPerson.iterrows():
            notThisPerson = (data['pid'] != thisPersonInst['pid'])
            aroundX = (abs(data['posX'] - thisPersonInst['posX']) < SQUARE_SIZE)
            aroundY = (abs(data['posY'] - thisPersonInst['posY']) < SQUARE_SIZE)
            time = abs(data['ts'] - thisPersonInst['ts'] < 30)
            aroundPeople_row_indices = data[aroundX & aroundY & time & notThisPerson].index
            aroundPeople = data.loc[aroundPeople_row_indices,:]
            aroundPeople['diffTs'] = abs(aroundPeople['ts'] - thisPersonInst['ts'])
            aroundPeople['time_rank'] = aroundPeople.groupby('pid', sort=False)['diffTs'].rank(ascending=True)
            #get only the closest time stamp of each person
            rank1 = aroundPeople['time_rank'] == 1
            rank1_row_indices = aroundPeople[rank1].index
            uAroundPeople = aroundPeople.loc[rank1_row_indices,:]
            #select the closest 10 people
            '''
            uAroundPeople['Xdist'] = (uAroundPeople['posX'] - thisPersonInst['posX'])
            uAroundPeople['Ydist'] = (uAroundPeople['posY'] - thisPersonInst['posY'])
            uAroundPeople['dist'] = abs(uAroundPeople['Xdist']) + abs(uAroundPeople['Ydist'])
            around = uAroundPeople.sort_values(by = ['dist'])
            nclosest = 10
            selectn = min(nclosest,around.shape[0])
            around = around.iloc[0:selectn]
            around = around[['Xdist','Ydist','velo','moAngle']]
            around['myVelo'] = thisPersonInst['velo']
            around['myFaceAngle'] = thisPersonInst['faceAngle']
            '''
            uAroundPeople['this_pid'] = thisPersonInst['pid']
            uAroundPeople['this_x'] = thisPersonInst['posX']
            uAroundPeople['this_y'] = thisPersonInst['posY']
            uAroundPeople['this_velo'] = thisPersonInst['velo']
            uAroundPeople['this_faceAngle'] = thisPersonInst['faceAngle']
            uAroundPeople['this_ts'] = thisPersonInst['ts']
            uAroundPeople['this_moAngle'] = thisPersonInst['moAngle']
            training.append(uAroundPeople)

            i+=1
            print(i)

            #fill upto 10 with default data if the length is less than 10
            
    return training
                   
def dist(p1,p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))
  
#creates a map of obstacles and free spaces 1 is obstacle and 0 is free space in the matrix returned      
def createMap(cords):
    #determine the size of the map (rectangle)
    [maxX,minX,maxY,minY] = [max(cords['posX']),min(cords['posX']),max(cords['posY']),min(cords['posY'])]
    cordIndices = cords.index
    normX = cords['posX'][cordIndices] - minX
    normY = cords['posY'][cordIndices] - minY
    normCords = pd.DataFrame(dict(x = normX,y=normY))
    normMaxX = max(normCords['x'])
    normMaxY = max(normCords['y'])
    maxcord = max(normMaxX,normMaxY)
    #Now seperate obstacles and free space
    GRIDSIZE = 1000  #grids would be GRIDSIZE * GIRDSIZE . this is in mm
    #get the dim of the sqare matrix
    dimX = math.ceil(normMaxX/GRIDSIZE)
    dimY = math.ceil(normMaxY/GRIDSIZE)

    mapMat = np.ones((dimY,dimX))
    #reduce the size of coordinates
    mapCords = (normCords/GRIDSIZE)
    mapCords = mapCords.apply(np.floor)
    mapCords = mapCords.drop_duplicates(subset = ['x','y'],keep = 'first')
    
    #mark the entries of mapMat where people were being
    for i in range(0,dimX): #x cordinates
        for j in range(0,dimY): # y cordinates
            #travel through mapCords and check
            print(i/dimX)
            if (((mapCords['x'] == i) & (mapCords['y'] == j)).any()):
                mapMat[j][i] = 0 # no obstacle here   
    #do not expand this scaled down map due to memory usage. Get the transformations on the go
    return [mapMat,GRIDSIZE]
     #figure(1)
     #imshow(mapMat, interpolation='nearest')
     #grid(True)
    
    
    
    


for raw in data:
    print(type(raw))
    
 for index, thisPersonInst in thisPerson.iterrows():
     print(thisPerson['posX'])

