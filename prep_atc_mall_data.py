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
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
#if you get SettingWithCopyWarning refer https://maxpowerwastaken.github.io/blog/pandas_view_vs_copy/


SQUARE_SIZE = 11000 #the size of the square the person looks around this is in mm
RES = 1000 # the length of a side of a single grid in the square this is in mm
GRIDSIZE = 1000  #grids would be GRIDSIZE * GIRDSIZE . this is in mm



data = pd.read_csv("/home/sleek_eagle/research/human_simulation/data/atc-20121114.csv",sep = ",",header=None,names =['ts','pid','posX','posY','posZ','velo','moAngle','faceAngle'])
cords = data[['posX','posY']]
cords.columns = ['x','y']

#number of uniqie people in the dataset
upids = data.pid.unique()
len(data.pid.unique())
aroundData = getAroundPeople(upids,2)
[mapMat,maxX,minX,maxY,minY,normCords] = createMap(cords)
features = getFeatureVectors(aroundData,mapMat,maxX,minX,maxY,minY)
[f,v] = [[],[]]
for i in features:
    [f_tmp,v_tmp] = unwrapFeatures(i)
    f.append(f_tmp)
    v.append(v_tmp)
    
v = pd.DataFrame(v)
v.columns = ['velo','angle']
v['sin'] = sin(v.angle)
v['cos'] = cos(v.angle)
v=v.drop(['angle'],axis = 1)
f = pd.DataFrame(f)
d = pd.concat([v,f],axis = 1)
d = d.sample(frac=1).reset_index(drop=True)
v = d[['velo','sin','cos']]
f = d.drop(['velo','sin','cos'],axis = 1)

#rescale the radian values. The values are -pi tp +pi.

TRAIN_RATIO = 0.6
n_data = f.shape[0]
v_train = v.loc[0:n_data*TRAIN_RATIO,]
f_train = f.loc[0:n_data*TRAIN_RATIO,]
v_test = v.loc[n_data*TRAIN_RATIO:n_data,]
f_test = f.loc[n_data*TRAIN_RATIO:n_data,]

#scale training data
scaler_f = preprocessing.StandardScaler().fit(f_train)
f_train_scaled = scaler_f.transform(f_train)
scaler_v = preprocessing.StandardScaler().fit(v_train)
v_train_scaled = scaler_v.transform(v_train)

#scale testing data
f_test_scaled = scaler_f.transform(f_test)
v_test_scaled = scaler_v.transform(v_test)

#training regressor (multiyayer NN)
reg = MLPRegressor(hidden_layer_sizes=(100,5,5), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
reg.fit(f_train_scaled,v_train_scaled)


#predict test data
pred = np.asmatrix(reg.predict(f_test_scaled))
#plot and analyze
v_test_scaled = pd.DataFrame(scaler_v.inverse_transform(v_test_scaled),columns = ['velo','sin','cos'])
pred = pd.DataFrame(scaler_v.inverse_transform(pred),columns = ['pred_velo','pred_sin','pred_cos'])
results = pd.concat([v_test_scaled,pred],axis = 1)


results = results.sort_values(by = ['velo']).reset_index()
plt.plot(results['velo'])
plt.plot(results['pred_velo'])
plt.show()

results = results.sort_values(by = ['sin']).reset_index()
plt.plot(results['sin'])
plt.plot(results['pred_sin'])
plt.show()

results = results.sort_values(by = ['cos']).reset_index()
plt.plot(results['cos'])
plt.plot(results['pred_cos'])
plt.show()


def scaleUp(scaled_values,maxvalue,minvalue):
    scaled_values = np.array(scaled_values)
    original_values = scaled_values*(maxvalue-minvalue) + minvalue
    return original_values

def scaleDown(values):
    values = np.array(values)
    maxvalue = max(values)
    minvalue = min(values)
    scaled = (values-minvalue)/(maxvalue-minvalue)
    return [scaled, maxvalue,minvalue]
    
    

# this wnwraps an instance of feature which consist of lists and matrices into a string
def unwrapFeatures(feature):
    f = []
    v = []
    for i in range(0,len(feature)):
        if (type(feature[i]) == list):
            for j in range(0,len(feature[i])):
                f.append(feature[i][j])
        if (type(feature[i]) == np.ndarray):
            for k in range(0,feature[i].shape[0]):
                for j in range(0,feature[i].shape[1]):
                    f.append(feature[i][k][j])
        if(type(feature[i]) == np.float64):
            v.append(feature[i])
    return [f,v]
            


def getFeatureVectors(aroundData,mapMat,maxX,minX,maxY,minY):
    LOOK_AHEAD = 50
    features = []
    for k in range(0,len(aroundData)-LOOK_AHEAD):
        # get the goal as the position this person will be in another 100 (or less) time steps
        this_pid = aroundData[k].iloc[0]['this_pid']
        if(((k+LOOK_AHEAD) >= len(aroundData)) and (aroundData[k+LOOK_AHEAD].iloc[0]['this_pid'] != this_pid)):
            break
        else:
            #relative position of goal wrt the current position
            goal_pos = [aroundData[k+LOOK_AHEAD].iloc[0]['this_x'] - aroundData[k].iloc[0]['this_x'],aroundData[k+LOOK_AHEAD].iloc[0]['this_y'] - aroundData[k].iloc[0]['this_y']]
    
        SQUARE_DIM = math.floor(SQUARE_SIZE/RES)
        mid_index = math.ceil(SQUARE_DIM/2)
        obs_around = np.zeros((SQUARE_DIM,SQUARE_DIM)) # matrix representing obstacles around
        for i in range(0,SQUARE_DIM):#x coordinates
            for j in range(0,SQUARE_DIM):#y coordinates
                #find what values in the mapMat this (j,i) corresponds to
                [mapx,mapy] = getMapValue(mapMat,maxX,minX,maxY,minY,(aroundData[k]['this_x'].iloc[0] + (i-mid_index)*RES),(aroundData[k]['this_y'].iloc[0] + (j-mid_index)*RES))
                #print(mapMat[mapy][mapx])
                #if the map has obstacle in this location, set to 1
                obs_around[i][j] = mapMat[mapy][mapx]
        #get people who are around this person
        xValues = aroundData[k]['posX']
        yValues = aroundData[k]['posY']
        thisX = aroundData[k]['this_x'].iloc[0]
        thisY = aroundData[k]['this_y'].iloc[0]
        people_around = np.zeros((SQUARE_DIM,SQUARE_DIM))
        people_velo = np.zeros((SQUARE_DIM,SQUARE_DIM))
        people_heading = np.zeros((SQUARE_DIM,SQUARE_DIM))
        for i in range(0,len(aroundData[k])):
            people_around[mid_index + math.floor((thisY - yValues.iloc[i])/RES)][mid_index + math.floor((thisX - xValues.iloc[i])/RES)] = 1
            people_velo[mid_index + math.floor((thisY - yValues.iloc[i])/RES)][mid_index + math.floor((thisX - xValues.iloc[i])/RES)] = aroundData[k]['velo'].iloc[i]
            people_heading[mid_index + math.floor((thisY - yValues.iloc[i])/RES)][mid_index + math.floor((thisX - xValues.iloc[i])/RES)] = aroundData[k]['moAngle'].iloc[i]

        feature = [goal_pos,obs_around,people_around,people_velo,people_heading,aroundData[k].iloc[0]['this_velo'],aroundData[k].iloc[0]['this_moAngle']]
        features.append(feature)
        
    return features
        
        
        
    
    
def getMapValue(mapMat,maxX,minX,maxY,minY,x,y):
    mapx = math.floor((x-minX)/GRIDSIZE)
    mapy = math.floor((y-minY)/GRIDSIZE)
    return [mapx, mapy]
    
    args = np.argwhere(mapMat==0)
    args = pd.DataFrame(args,columns = ['x','y'])
    plt.scatter(mapCords['x'],mapCords['y'])
    
    
               
def getAroundPeople(upids,personLimit):
    aroundData = []
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
            aroundX = (abs(data['posX'] - thisPersonInst['posX']) < ((SQUARE_SIZE/RES+1)/2-1)*RES)
            aroundY = (abs(data['posY'] - thisPersonInst['posY']) < ((SQUARE_SIZE/RES+1)/2-1)*RES)
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
            aroundData.append(uAroundPeople)

            i+=1
            print(i)

            #fill upto 10 with default data if the length is less than 10
            
    return aroundData
                   
def dist(p1,p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))

def scaleDownCordinates(cords,minX,minY):
    normX = (cords['x'] -minX)/GRIDSIZE
    normY = (cords['y'] -minY)/GRIDSIZE
    normCords = pd.DataFrame(dict(x = normX,y=normY))
    normCords=normCords.apply(np.floor)
    normCords = normCords.drop_duplicates()
    return normCords


    
  
#creates a map of obstacles and free spaces 1 is obstacle and 0 is free space in the matrix returned      
def createMap(cords):
    #determine the size of the map (rectangle)
    [maxX,minX,maxY,minY] = [max(cords['x']),min(cords['x']),max(cords['y']),min(cords['y'])]
    normCords = scaleDownCordinates(cords,minX,minY)
    normCords = normCords.reset_index(drop = True)
    normMaxX = max(normCords['x'])
    normMaxY = max(normCords['y'])
    #Now seperate obstacles and free space
    #get the dim of the sqare matrix
    dimX = math.ceil(normMaxX)+1
    dimY = math.ceil(normMaxY)+1
    
    mapMat = np.ones((dimY,dimX))
    
    #mark the entries of mapMat where people were being
    for index, row in normCords.iterrows():
        mapMat[int(row['y']),int(row['x'])]=0
    
    #do not expand this scaled down map due to memory usage. Get the transformations on the go
    return [mapMat,maxX,minX,maxY,minY,normCords]
    


