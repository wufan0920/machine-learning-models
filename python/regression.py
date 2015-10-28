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

def analytic_solution(X,Y):
    M = nu.mat(add_bias(X))
    #analytic solution
    W = (M.T*M).I*M.T*Y
    predict = nu.array(M*W)
    return predict

def normalization(X):
    m = nu.array(X)
    means = m.mean(axis=0)
    stds = m.std(axis=0)
    print m
    print means

    means[-1] = 0.0
    stds[-1] = 1.0

    m = m - nu.tile(means,(len(m),1))
    m = m/nu.tile(stds,(len(m),1))
    return m

def gradient_descent(X,Y):
    length = len(X)
    m = add_bias(X)
    m = normalization(m)
    M = nu.mat(m)

    iters = 100
    alphas = [0.01, 0.03, 0.1, 0.3, 1, 1.27]
    styles = ['b', 'r', 'g', 'k', 'b--', 'r--']

    for alpha in alphas:
        W = nu.mat(nu.zeros((M.shape[1],1)))
        J = []
        x_axis = range(50)
        for itr in range(iters):
            grad = (1.0/length) * (M.T * ((M*W)-Y))
            lost = (1.0/2.0*length) * ((M*W)-Y).T * ((M*W)-Y)
            J.append(float(lost))
            W = W - alpha * grad
        plt.plot(x_axis,J[:50],styles[alphas.index(alpha)]) 

    predict = nu.array(M*W)
    return predict

if __name__=='__main__':
    X,Y = load_data('dataset/ex3Data/ex3x.dat','dataset/ex3Data/ex3y.dat')

    #analytic solution
    #predict = analytic_solution(X,Y) 

    #gradient descent solution
    predict = gradient_descent(X,Y) 

    plt.plot(X,predict,'b-')
    plt.xlabel('Num of iterations')
    plt.ylabel('Cost Function')
    plt.show()
