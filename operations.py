#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:46:37 2018

@author: sleek_eagle
"""

import numpy as np
import math

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
        if(i >= len(feature)-2):
            v.append(feature[i])
    return [f,v]

def getAngle(sin_val,cos_val):
    quad = [0,0,0,0]
    if (sin_val > 0):
        quad[0] = 1
        quad[1] = 1
        quad[2] = 0
        quad[3] = 0
    else:
        quad[0] = 0
        quad[1] = 0
        quad[2] = 1
        quad[3] = 1
        
    if (cos_val > 0):
        quad[0] = 1*quad[0]
        quad[1] = 0*quad[1]
        quad[2] = 0*quad[2]
        quad[3] = 1*quad[3]
    else:
        quad[0] = 0*quad[0]
        quad[1] = 1*quad[1]
        quad[2] = 1*quad[2]
        quad[3] = 0*quad[3]

    if(quad[0] == 1):
        sangle = math.asin(sin_val)
        cangle = math.acos(cos_val)
    if(quad[1] == 1):
        sangle = (math.pi/2 - math.asin(sin_val)) + math.pi/2
        cangle = math.acos(cos_val)
    elif(quad[2] == 1):
        sangle = -1 * math.asin(sin_val) + math.pi
        cangle = math.pi - math.acos(cos_val) + math.pi
    elif(quad[3] == 1):
        sangle = 2*math.pi + math.asin(sin_val)
        cangle = 2*math.pi - math.acos(cos_val)
        
    angle = (sangle + cangle)/2
    return angle
        
        

        
    