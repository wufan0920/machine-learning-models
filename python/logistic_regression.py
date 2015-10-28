#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as nu

def load_data(pathX,pathY):
    matrixX = [] 
    matrixY = [] 
    dataX = file(pathX)
    dataY = file(pathY)

    for line in dataX.readlines():
        row = []
        x = line.strip().split()
        for column in x:
            tmp = float(column)
            row.append(tmp)
        matrixX.append(row)

    for line in dataY.readlines():
        row = []
        y = line.strip().split()
        for column in y:
            tmp = float(column)
            row.append(tmp)
        matrixY.append(row)
    
    return matrixX, matrixY

def add_bias(inputs):
    for i in range(len(inputs)):
        inputs[i].append(1)
    return inputs

def print_training_data(X,Y):
    x1p = []
    x2p = []
    x1f = []
    x2f = []
    for i,value in enumerate(Y):
        if value[0] == 0:
            x1f.append(X[i][0])
            x2f.append(X[i][1])
        else:
            x1p.append(X[i][0])
            x2p.append(X[i][1])
         
    plt.plot(x1f,x2f,'+')
    plt.plot(x1p,x2p,'bo')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')

def sigmoid(inputs):
    z = nu.mat(nu.zeros(inputs.shape))
    return 1.0/(1.0+nu.exp(z-inputs))

def diag_mat(h):
    a = nu.array(h)
    diag = nu.diag(a.reshape([a.shape[0],]))
    return diag

def newton_method(X,Y):
    length = len(X)
    iters = 7
    M = nu.mat(add_bias(X))
    W = nu.mat(nu.zeros((M.shape[1],1)))

    for itr in range(iters):
        h = sigmoid(M*W)
        grad = (1.0/length)*(M.T*(h-Y))
        diag = diag_mat(h)
        diag2 = diag_mat(nu.ones(h.shape) - h)

        H = (1.0/length)*M.T*diag*diag2*M

        W = W - (H.I*grad)

    print W


if __name__=='__main__':
    X,Y = load_data('dataset/ex4Data/ex4x.dat','dataset/ex4Data/ex4y.dat')

    print_training_data(X,Y) 
    newton_method(X,Y)
    plt.show()
