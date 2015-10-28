#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
import math
import functools

def load_data(pathX):
    matrixX = [] 
    dataX = file(pathX)

    for line in dataX.readlines():
        row = []
        x = line.strip().split()
        for column in x:
            tmp = float(column)
            row.append(tmp)
        matrixX.append(row)
    
    return matrixX

if __name__=='__main__':
    W = load_data('out.txt')
    W = np.array(W)
    print W.shape
    W1 = np.reshape(W[:25*64,], (25, 64))

    #plot the weight
    fig = plt.figure(2)

    for index in range(25):
        weight = W1[index,:]
        weight = np.reshape(weight,(8,8))
        ax = fig.add_subplot(5,5,1+index)
        ax.imshow(weight,mpl.cm.gray)

    plt.show()
