import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import nnet
import softmax
import matplotlib.pyplot as plt
import matplotlib as mpl

import time

def load_data(pathX):
    matrixX = [] 
    dataX = file(pathX)

    for line in dataX.readlines():
        row = []
        x = line.strip().split(',')
        for column in x:
            tmp = float(column)
            row.append(tmp)
        matrixX.append(row)
    
    return matrixX

def savedata(data,filename):
    data = str(data.tolist())
    #save_file = open('weight','w')
    save_file = open(filename,'w')
    save_file.write(data)
    save_file.flush()
    save_file.close()

if __name__=='__main__':
    #public variables
    inputSize = 28*28
    numClasses = 10
    hiddenSizeL1 = 200
    #hiddenSizeL2 = 200
    hiddenSizeL2 = 100
    sparsityParam = 0.1
    beta = 3
    lmd = 0.001
    #alpha = 0.03
    alpha = 0.04

    #step1 load mnist training & tesing data
    image = sio.loadmat('dataset/mnist_dataset/mnist_train.mat')
    label = sio.loadmat('dataset/mnist_dataset/mnist_train_labels.mat')
    data = image['mnist_train']
    training_labels = label['mnist_train_labels']
    training_data = data.transpose()
    #testing data
    image = sio.loadmat('dataset/mnist_dataset/mnist_test.mat')
    label = sio.loadmat('dataset/mnist_dataset/mnist_test_labels.mat')
    test_data = image['mnist_test']
    test_labels = label['mnist_test_labels']
    test_data = test_data.transpose()

    start = time.time()
    
    #step2 L1 feature learning using sparse autoencoder
    #TODO move normalization to miscellaneous
    training_data = nnet.normalization(training_data)
    #W = nnet.sparseAutoencoder(inputSize,hiddenSizeL1,sparsityParam,lmd,beta,alpha,training_data,iters=500)
    W = load_data('weightL1')
    W = np.array(W)
    W = W.transpose()
    #savedata(W,'weightL1')
    W11 = np.reshape(W[:hiddenSizeL1*inputSize,], (hiddenSizeL1, inputSize))
    b11 = np.reshape(W[2*hiddenSizeL1*inputSize:2*hiddenSizeL1*inputSize+hiddenSizeL1,],(hiddenSizeL1,1))

    #step3 L2 feature learning using sparse autoencoder
    training_a1 = nnet.sigmoid(W11.dot(training_data)+b11)
    #W = nnet.sparseAutoencoder(hiddenSizeL1,hiddenSizeL2,sparsityParam,lmd,beta,0.009,training_a1,iters=500)
    W = load_data('weightL2')
    W = np.array(W)
    W = W.transpose()
    #savedata(W,'weightL2')
    W21 = np.reshape(W[:hiddenSizeL2*hiddenSizeL1,], (hiddenSizeL2, hiddenSizeL1))
    b21 = np.reshape(W[2*hiddenSizeL2*hiddenSizeL1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2,],(hiddenSizeL2,1))

    #step4 plot the learned feature
    #fig = plt.figure(2)

    #for index in range(hiddenSizeL1/10):
    #    weight = W11[index,:]
    #    weight = np.reshape(weight,(28,28))
    #    #print weight.shape
    #    ax = fig.add_subplot(5,4,1+index)
    #    ax.imshow(weight,mpl.cm.gray)

    #plt.show()


    #step 6 softmax regression
    test_data = nnet.normalization(test_data)

    train_a1 = nnet.sigmoid(W11.dot(training_data)+b11)
    train_a2 = nnet.sigmoid(W21.dot(train_a1)+b21)
    #test_a1 = nnet.sigmoid(W11.dot(training_data)+b11)
    test_a1 = nnet.sigmoid(W11.dot(test_data)+b11)
    test_a2 = nnet.sigmoid(W21.dot(test_a1)+b21)

    W = softmax.softmax_regression(hiddenSizeL2,numClasses,0,train_a2,training_labels,800,a=0.7)
    #W = softmax.softmax_regression(hiddenSizeL1,numClasses,lmd,train_a1,training_labels,100)

    #step 7 testing
    theta = W.reshape((numClasses, hiddenSizeL2))
    predict = (theta.dot(test_a2)).argmax(0)
    #theta = W.reshape((numClasses, hiddenSizeL1))
    #predict = (theta.dot(test_a1)).argmax(0)
    print predict
    print test_labels.flatten()
    accuracy = (predict == test_labels.flatten())
    #accuracy = (predict == training_labels.flatten())
    print 'Accuracy:',accuracy.mean()
    end = time.time()
    print 'time cost: ', end-start
    print 'done'
