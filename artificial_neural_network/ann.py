# Machine Learning
# Homework 5
# 5-layer Artificial Neural Network
# Madison Wang

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

def one_epoch(inputs, labels, num, weights, eta):
    length = len(inputs)
    for c in range(length):
        layers = [np.append(inputs[c], 1)]
        for i in range(num-1):
            layers.append(1 / (1 + np.exp(0 - np.dot(layers[i], weights[i]))))
        ey = np.zeros((1, 10))[0]
        ey[int(labels[c])] = 1
        diff = layers[-1] - ey
        err = np.dot(diff, diff)
        
        deltas = [0 for k in range(num)]
        #deltas for the last layer
        deltas[-1] = diff * layers[-1] * (1 - layers[-1])
        #sum w_{t->u}delta_u
        weighted_sum_delta = weights[-1] * deltas[-1]
        #weights[-1] -= eta * np.matmul(np.matrix(layers[-2]).T, deltas[-1])
        for j in range(num - 2, -1, -1):
            deltas[j] = layers[j] * (1 - layers[j]) * np.dot(weights[j], deltas[j + 1])
            weights[j] -= eta * np.matmul(np.matrix(layers[j]).T, np.matrix(deltas[j + 1]))
    return weights

def predict(inputs, labels, weights, num, calculate_error):
    length = len(inputs)
    err = 0
    pre = []
    for c in range(length):
        layers = np.append(inputs[c], 1)
        for i in range(num-1):
            layers = 1 / (1 + np.exp(0 - np.dot(layers, weights[i])))
        predicted = np.argmax(layers)
        if calculate_error:
            err += (predicted != labels[c]) / length
        else :
            pre.append(np.argmax(layers))
    
    if calculate_error:
        return err
    else:
        return pre     

            
def go():
    train_digit_x = np.loadtxt("TrainDigitX.csv.gz", delimiter=',')
    train_digit_y = np.loadtxt("TrainDigitY.csv.gz", delimiter=',')
    test_digit_x = np.loadtxt("TestDigitX.csv.gz", delimiter=',')
    test_digit_x2 = np.loadtxt("TestDigitX2.csv.gz", delimiter=',')
    #test_digit_y = np.loadtxt("TestDigitY.csv.gz", delimiter=',')
    
    n_epochs = 30
    eta = 0.125
    dims = [785, 257, 129, 65, 10]
    #Weights going from layer 1 to layer 2
    #Column j is a set of weights from layer 1 to the jth neuron in layer 2
    ws12 = np.random.rand(dims[0], dims[1]) - 0.5
    #Weights going from layer 2 to layer 3
    ws23 = np.random.rand(dims[1], dims[2]) - 0.5
    ws34 = np.random.rand(dims[2], dims[3]) - 0.5
    ws45 = np.random.rand(dims[3], dims[4]) - 0.5
    weights = [ws12, ws23, ws34, ws45]
    count = 0
    while count < n_epochs:    
        weights = one_epoch(train_digit_x, train_digit_y, 5, weights, eta)
        count += 1
    
    pre1 = predict(test_digit_x, None, weights, 5, False)
    pre2 = predict(test_digit_x2, None, weights, 5, False)
    length1 = len(pre1)
    length2 = len(pre2)
    
    file = open("predict_x.txt", "w")
    for pre in pre1[:length1-1]:
        file.write("{},".format(pre))
    file.write("{}".format(pre1[-1]))
    file.close()
    
    file = open("predict_x2.txt", "w")
    for pre in pre2[:length2-1]:
        file.write("{},".format(pre))
    file.write("{}".format(pre2[-1]))
    file.close()
    
    return pre1, pre2
    
go()   