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
from sklearn.externals import joblib
from operations import unwrapFeatures
from operations import getAngle
from operations import getAngleFromCoords
from scipy.stats.stats import pearsonr   




#if you get SettingWithCopyWarning refer https://maxpowerwastaken.github.io/blog/pandas_view_vs_copy/


SQUARE_SIZE = 5000 #the size of the square the person looks around this is in mm
GRIDSIZE = 1000  #grids would be GRIDSIZE * GIRDSIZE . this is in mm



data = pd.read_csv("/home/sleek_eagle/research/human_simulation/data/atc-20121114.csv",sep = ",",header=None,names =['ts','pid','posX','posY','posZ','velo','moAngle','faceAngle'])


#make minus angles positive
neg = data['moAngle'] < 0
data.loc[neg,'moAngle'] = math.pi*2 + data['moAngle']
cords = data[['posX','posY']]
cords.columns = ['x','y']

data['velo_x'] = data['velo'] * data['moAngle'].apply(lambda x:math.cos(x))
data['velo_y'] = data['velo'] * data['moAngle'].apply(lambda x:math.sin(x))
data['velo_sum'] = abs(data['velo_x']) + abs(data['velo_y'])
data['velo_x'] = data['velo_x']/data['velo_sum']
data['velo_y'] = data['velo_y']/data['velo_sum']



#scale data here
'''
maxvelo = max(data['velo'])
data['velo'] = data['velo']/maxvelo
'''
#data['moAngle'] = data['moAngle']/(math.pi*2)
#data['faceAngle'] = data['faceAngle']/(math.pi*2)


#number of uniqie people in the dataset
upids = data.pid.unique()
len(data.pid.unique())
[aroundData,moangles] = getAroundPeople(upids,20)
[mapMat,maxX,minX,maxY,minY,normCords] = createMap(cords)
plt.matshow(mapMat)
features = getFeatureVectors_v2(aroundData,mapMat,maxX,minX,maxY,minY)
[f,v] = [[],[]]
for i in features:
    [f_tmp,v_tmp] = unwrapFeatures(i)
    f.append(f_tmp)
    v.append(v_tmp)
    
v = pd.DataFrame(v)
v.columns = ['velo_x','velo_y']
#neg = v['angle'] < 0
#v.loc[neg,'angle'] = math.pi*2 + v['angle']
#v['sin'] = sin(v.angle)
#v['cos'] = cos(v.angle)
#v=v.drop(['angle'],axis = 1)
f = pd.DataFrame(f)


#test
'''
ma = []
for i in range(0,len(moangles)):
    ma.append(moangles[i].iloc[0])
'''
    
#f = f[[0,1]]
d = pd.concat([v,f],axis = 1)
#data set if unbalanced on angle (more positive angles. so remove some of them)
'''
d = d.sort_values(by = ['angle']).reset_index(drop = True)
neg_angles = len(d[d['angle'] < 0])
pos_angles = len(d[d['angle'] > 0])
dif = abs(pos_angles - neg_angles)
if(neg_angles < pos_angles):
     selected = d.loc[0:(len(d)-dif),:]
else:
    selected = d.loc[dif:(len(d)-1),:]
    
d = selected
'''
#rescale velo to -1 +1


d = d.sample(frac=1).reset_index(drop=True)
v = d[['velo_x','velo_y']]
f = d.drop(['velo_x','velo_y'],axis = 1)
v = pd.DataFrame(v)

#creat sin and cos columns for both f and v
'''
v['sin'] = sin(v['angle'])
v['cos'] = cos(v['angle'])
f['sin'] = sin(f[0])
f['cos'] = cos(f[0]) 
v = v.drop(['angle'],axis = 1)
f = f.drop([0],axis=1)
cols = f.columns.tolist()
cols = cols[-2:] + cols[:-2]
f = f[cols]
'''

TRAIN_RATIO = 0.6
n_data = f.shape[0]
v_train = v.loc[0:n_data*TRAIN_RATIO,]
f_train = f.loc[0:n_data*TRAIN_RATIO,]
v_test = v.loc[n_data*TRAIN_RATIO:n_data,]
f_test = f.loc[n_data*TRAIN_RATIO:n_data,]


#scale training data
'''
scaler_f = preprocessing.StandardScaler().fit(f_train)
f_train_scaled = scaler_f.transform(f_train)
scaler_v = preprocessing.StandardScaler().fit(v_train)
v_train_scaled = scaler_v.transform(v_train)

#scale testing data
f_test_scaled = scaler_f.transform(f_test)
v_test_scaled = scaler_v.transform(v_test)


#save scalers to file
joblib.dump(scaler_f, 'scaler_f.pkl') 
joblib.dump(scaler_v, 'scaler_v.pkl') 
'''





#training regressor (multiyayer NN)
reg = MLPRegressor(hidden_layer_sizes=(300,100), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
reg.fit(f_train, v_train)
joblib.dump(reg, 'NNmodel.pkl') 

#test


#predict test data
pred = pd.DataFrame(np.asmatrix(reg.predict(f_test)))
pred.columns = ['pred_velo_x','pred_velo_y']
#simple rescaling
pred['velo_sum'] = abs(pred['pred_velo_x']) + abs(pred['pred_velo_y'])
pred['pred_velo_x'] = pred['pred_velo_x']/pred['velo_sum']
data['pred_velo_y'] = pred['pred_velo_y']/pred['velo_sum']

#plot and analyze
'''
v_test_scaled_back = pd.DataFrame(scaler_v.inverse_transform(v_test_scaled),columns = ['velo','sin','cos'])
pred = pd.DataFrame(scaler_v.inverse_transform(pred),columns = ['pred_velo','pred_sin','pred_cos'])
'''

results = pd.concat([v_test.reset_index(),pred],axis = 1)
results = results.drop(['index'],axis = 1)

'''
results.columns = ['velo','sin','cos','pred_velo','pred_sin','pred_cos']
#rescale sin and cos
results['angle'] = results.apply(getDFangle1,axis=1)
results['pred_angle'] = results.apply(getDFangle2,axis=1)
'''

def getDFangle1(row):
    #print("in here!")
    return getAngle(row['sin'],row['cos'])
def getDFangle2(row):
    #print("in here!")
    return getAngle(row['pred_sin'],row['pred_cos'])


#velo_x analyze
results = results.sort_values(by = ['velo_x']).reset_index()
fig = plt.figure()
predplot = plt.plot(results['pred_velo_x'],label = 'predicted')
actualplot = plt.plot(results['velo_x'], label = 'actual')
plt.xlabel('Instances', fontsize=24)
plt.ylabel('velocity_x', fontsize=24)
plt.legend(bbox_to_anchor=(0.1, 0.99), loc=1, borderaxespad=0.)
plt.show()


pearsonr(results['velo_x'],results['pred_velo_x'])
results['error']  = abs(results['pred_velo_x'] - results['velo_x'])
plt.plot(results['error'] )
plt.show()

#velo_y analyze
results = results.sort_values(by = ['velo_y']).reset_index()
fig = plt.figure()
predplot = plt.plot(results['pred_velo_y'],label = 'predicted')
actualplot = plt.plot(results['velo_y'], label = 'actual')
plt.xlabel('Instances', fontsize=24)
plt.ylabel('velocity_y', fontsize=24)
plt.legend(bbox_to_anchor=(0.1, 0.99), loc=1, borderaxespad=0.)
plt.show()


pearsonr(results['velo_y'],results['pred_velo_y'])
results['error']  = abs(results['pred_velo_y'] - results['velo_y'])
plt.plot(results['error'] )
plt.show()

#test
f = [[-0.5,-0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
reg.predict(f)

'''
results = results.sort_values(by = ['sin']).reset_index()
plt.plot(results['sin'])
plt.plot(results['pred_sin'])
plt.show()

results = results.sort_values(by = ['cos']).reset_index()
plt.plot(results['cos'])
plt.plot(results['pred_cos'])
plt.show()


results = results.sort_values(by = ['angle']).reset_index()
plt.plot(results['pred_angle']*180/pi,label = 'predicted')
plt.plot(results['angle']*180/pi,label = 'actual')
plt.xlabel('Instances', fontsize=16)
plt.ylabel('angle', fontsize=16)
plt.legend(bbox_to_anchor=(0.9, 1), loc=1, borderaxespad=0.)
results['error'] = abs(results['angle'] - results['pred_angle'])
plt.show()
pearsonr(results['angle'],results['pred_angle'])
plt.plot(abs(results['angle'] - results['pred_angle']))
analyse = pd.DataFrame(results[['angle','error']])
analyse['angle'] = round(analyse['angle'],1)
ana = analyse.groupby(['angle']).mean().reset_index()
plt.plot(ana['angle'],ana['error'])
plt.xlabel('angle', fontsize=16)
plt.ylabel('average error', fontsize=16)
'''



#test
f_test_scaled = np.asmatrix(f_test_scaled)
for i in range(0,len(aroundData)):
    if((aroundData[i]['this_velo'].iloc[0] < 0.0)):
        print("negative!")
        
# are around data sequencial in time ?
pid = []
for i in range(0,len(aroundData)):
    pid.append(aroundData[i]['this_pid'].iloc[0])
pid = pd.DataFrame(pid)
plt.plot(ts)# yes it is. no need of sorting on this_ts

array = []
for i in range(0,len(features)):
    quad = -1
    x = features[i][0][0]
    y = features[i][0][1]
    if((x > 0) and (y > 0)): quad = 1
    if((x < 0) and (y > 0)): quad = 2
    if((x < 0) and  (y < 0)): quad = 3
    if((x > 0) and (y < 0)): quad = 4
    array.append([x,y,quad,features[i][len(features[i])-1]*180/math.pi])
array = pd.DataFrame(array)
array.columns = ['dx','dy','quad','angle']
array[0]
    

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
    
    
'''
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
'''            


def getFeatureVectors_v2(aroundData,mapMat,maxX,minX,maxY,minY):
    LOOK_AHEAD = 100
    features = []
    for k in range(0,len(aroundData)-LOOK_AHEAD):
        # get the goal as the position this person will be in another 100 (or less) time steps
        this_pid = aroundData[k].iloc[0]['this_pid']
        print(k)
        if(((k+LOOK_AHEAD) >= len(aroundData))):
            break
        if((aroundData[k+LOOK_AHEAD].iloc[0]['this_pid'] != this_pid)):
            continue
        else:
            #relative position of goal wrt the current position
            goal_pos = [aroundData[k+LOOK_AHEAD].iloc[0]['this_x'] - aroundData[k].iloc[0]['this_x'],aroundData[k+LOOK_AHEAD].iloc[0]['this_y'] - aroundData[k].iloc[0]['this_y']]    
            #normalize this
            goal_sum = abs(goal_pos[0]) + abs(goal_pos[1])
            goal_pos[0] /= goal_sum
            goal_pos[1] /= goal_sum
        SQUARE_DIM = math.floor(SQUARE_SIZE/GRIDSIZE)
        mid_index = math.ceil(SQUARE_DIM/2)
        obs_around = np.zeros((SQUARE_DIM,SQUARE_DIM)) # matrix representing obstacles around
        for i in range(0,SQUARE_DIM):#x coordinates
            for j in range(0,SQUARE_DIM):#y coordinates
                #find what values in the mapMat this (j,i) corresponds to
                [mapx,mapy] = getMapValue(mapMat,maxX,minX,maxY,minY,(aroundData[k]['this_x'].iloc[0] + (i-mid_index)*GRIDSIZE),(aroundData[k]['this_y'].iloc[0] + (j-mid_index)*GRIDSIZE))
                #print(mapMat[mapy][mapx])
                #if the map has obstacle in this location, set to 1
                obs_around[i][j] = mapMat[mapy][mapx]
        #get people who are around this person
        xValues = aroundData[k]['posX']
        yValues = aroundData[k]['posY']
        thisX = aroundData[k]['this_x'].iloc[0]
        thisY = aroundData[k]['this_y'].iloc[0]
        people_around = np.zeros((SQUARE_DIM,SQUARE_DIM))
        people_velo_x = np.zeros((SQUARE_DIM,SQUARE_DIM))
        people_velo_y = np.zeros((SQUARE_DIM,SQUARE_DIM))
        for i in range(0,len(aroundData[k])):
            people_around[mid_index + math.floor((thisY - yValues.iloc[i])/GRIDSIZE)][mid_index + math.floor((thisX - xValues.iloc[i])/GRIDSIZE)] = 1
            people_velo_x[mid_index + math.floor((thisY - yValues.iloc[i])/GRIDSIZE)][mid_index + math.floor((thisX - xValues.iloc[i])/GRIDSIZE)] = aroundData[k]['velo_x'].iloc[i]
            people_velo_y[mid_index + math.floor((thisY - yValues.iloc[i])/GRIDSIZE)][mid_index + math.floor((thisX - xValues.iloc[i])/GRIDSIZE)] = aroundData[k]['velo_y'].iloc[i]


        #combine matrices
        around_mat =  (np.logical_or(people_around,obs_around)).astype(int)
        
        

        feature = [goal_pos[0],goal_pos[1],around_mat,people_velo_x,people_velo_y,aroundData[k].iloc[0]['this_velo_x'],aroundData[k].iloc[0]['this_velo_y']]
        features.append(feature)
        
    return features
        

def getFeatureVectors(aroundData,mapMat,maxX,minX,maxY,minY):
    LOOK_AHEAD = 100
    features = []
    for k in range(0,len(aroundData)-LOOK_AHEAD):
        # get the goal as the position this person will be in another 100 (or less) time steps
        this_pid = aroundData[k].iloc[0]['this_pid']
        print(k)
        if(((k+LOOK_AHEAD) >= len(aroundData))):
            break
        if((aroundData[k+LOOK_AHEAD].iloc[0]['this_pid'] != this_pid)):
            continue
        else:
            #relative position of goal wrt the current position
            goal_pos = [aroundData[k+LOOK_AHEAD].iloc[0]['this_x'] - aroundData[k].iloc[0]['this_x'],aroundData[k+LOOK_AHEAD].iloc[0]['this_y'] - aroundData[k].iloc[0]['this_y']]    
            angle = getAngleFromCoords(goal_pos[0],goal_pos[1])
            goal_pos = angle
        SQUARE_DIM = math.floor(SQUARE_SIZE/GRIDSIZE)
        mid_index = math.ceil(SQUARE_DIM/2)
        obs_around = np.zeros((SQUARE_DIM,SQUARE_DIM)) # matrix representing obstacles around
        for i in range(0,SQUARE_DIM):#x coordinates
            for j in range(0,SQUARE_DIM):#y coordinates
                #find what values in the mapMat this (j,i) corresponds to
                [mapx,mapy] = getMapValue(mapMat,maxX,minX,maxY,minY,(aroundData[k]['this_x'].iloc[0] + (i-mid_index)*GRIDSIZE),(aroundData[k]['this_y'].iloc[0] + (j-mid_index)*GRIDSIZE))
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
            people_around[mid_index + math.floor((thisY - yValues.iloc[i])/GRIDSIZE)][mid_index + math.floor((thisX - xValues.iloc[i])/GRIDSIZE)] = 1
            people_velo[mid_index + math.floor((thisY - yValues.iloc[i])/GRIDSIZE)][mid_index + math.floor((thisX - xValues.iloc[i])/GRIDSIZE)] = aroundData[k]['velo'].iloc[i]
            people_heading[mid_index + math.floor((thisY - yValues.iloc[i])/GRIDSIZE)][mid_index + math.floor((thisX - xValues.iloc[i])/GRIDSIZE)] = aroundData[k]['moAngle'].iloc[i]

        feature = [math.sin(goal_pos),math.cos(goal_pos),obs_around,people_around,people_velo,people_heading,aroundData[k].iloc[0]['this_velo'],math.sin(aroundData[k].iloc[0]['this_moAngle']),math.cos(aroundData[k].iloc[0]['this_moAngle'])]
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
    moangles = []
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
            aroundX = (abs(data['posX'] - thisPersonInst['posX']) < ((SQUARE_SIZE/GRIDSIZE+1)/2-1)*GRIDSIZE)
            aroundY = (abs(data['posY'] - thisPersonInst['posY']) < ((SQUARE_SIZE/GRIDSIZE+1)/2-1)*GRIDSIZE)
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
            uAroundPeople['this_velo_x'] = thisPersonInst['velo_x']
            uAroundPeople['this_velo_y'] = thisPersonInst['velo_y']

            if(uAroundPeople.shape[0] > 0):
                aroundData.append(uAroundPeople)
                moangles.append(uAroundPeople['this_moAngle'])
            i+=1
            print(i)

            #fill upto 10 with default data if the length is less than 10
            
    return [aroundData,moangles]
                   
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
    


