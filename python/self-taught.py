import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import nnet
import softmax
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__=='__main__':
    #public variables
    inputSize = 28*28
    numLabels = 5
    hiddenSize = 200
    sparsityParam = 0.1
    beta = 3
    lmd = 0.001
    alpha = 0.04

    #step1 load mnist training data
    image = sio.loadmat('dataset/mnist_dataset/mnist_train.mat')
    label = sio.loadmat('dataset/mnist_dataset/mnist_train_labels.mat')
    data = image['mnist_train']
    labels = label['mnist_train_labels']
    data = data.transpose()

    #for test 
    image = sio.loadmat('dataset/mnist_dataset/mnist_test.mat')
    label = sio.loadmat('dataset/mnist_dataset/mnist_test_labels.mat')
    test_data = image['mnist_test']
    test_labels = label['mnist_test_labels']
    test_data = test_data.transpose()

    test_set = np.nonzero(test_labels<=4)
    test_labels = test_labels[test_set[0].flatten()]
    test_data = test_data[:,test_set[0].flatten()]

    #step2 divide training set into feature learning set & supervised training set
    labeled_set = np.nonzero(labels>=5)
    unlabeled_set = np.nonzero(labels<=4)
    
    labeled_dataset = data[:,(labels<=4).flatten()]
    labeled_labelset = labels[(labels<=4).flatten()]
    unlabeled_dataset = data[:,(labels>=5).flatten()]

    unlabeled_dataset = unlabeled_dataset[:,:unlabeled_dataset.shape[1]/3]
    print unlabeled_dataset.shape

    #step3 feature learning using sparse autoencoder
    #TODO move normalization to miscellaneous
    unlabeled_dataset = nnet.normalization(unlabeled_dataset)
    W = nnet.sparseAutoencoder(inputSize,hiddenSize,sparsityParam,lmd,beta,alpha,unlabeled_dataset)
    W1 = np.reshape(W[:hiddenSize*inputSize,], (hiddenSize, inputSize))
    b1 = np.reshape(W[2*hiddenSize*inputSize:2*hiddenSize*inputSize+hiddenSize,],(hiddenSize,1))

    #step4 plot the learned feature
    fig = plt.figure(2)

    for index in range(hiddenSize/10):
        weight = W1[index,:]
        weight = np.reshape(weight,(28,28))
        ax = fig.add_subplot(5,4,1+index)
        ax.imshow(weight,mpl.cm.gray)

    plt.show()

    #step5 extract features from test & training data
    #TODO move sigmoid to miscellaneous
    labeled_dataset = nnet.normalization(labeled_dataset)
    test_data = nnet.normalization(test_data)
    train_a1 = nnet.sigmoid(W1.dot(labeled_dataset)+b1)
    test_a1 = nnet.sigmoid(W1.dot(test_data)+b1)

    #step 6 softmax regression
    W = softmax.softmax_regression(hiddenSize,numLabels,lmd,train_a1,labeled_labelset,100)

    #step 7 testing
    theta = W.reshape((numLabels, hiddenSize))
    predict = (theta.dot(test_a1)).argmax(0)
    print predict
    print test_labels.flatten()
    accuracy = (predict == test_labels.flatten())
    print 'Accuracy:',accuracy.mean()
    print 'done'
    
    
