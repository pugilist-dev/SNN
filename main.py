#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 22:59:27 2020

@author: rajiv
"""

import numpy as np
from scipy.io import loadmat
from lif_neuron import lif_neuron
import matplotlib.pyplot as plt
import math

dataset = loadmat('mnist.mat')

M = dataset['M'] # Number of classes
N = dataset['N'] # number of inputs 28*28 = 784
m_train = dataset['m_train'] # 60000 training examples
m_test = dataset['m_test'] # 10000 testing examples
U_train = dataset['U_train'] # Training data
U_test = dataset['U_test'] # testing data
Y_train = dataset['Y_train'] # Training labels
Y_test = dataset['Y_test'] # Testing labels
train_labels = dataset['train_labels'] # integer values of the train labels
test_labels = dataset['test_labels'] # Integer values of the test labels

layers = 2
neurons_per_layer = 2

# input 1 of the XOR gate
input1 = np.zeros(200)
input1 = np.append(input1, np.ones(200))
input1 = np.append(input1, np.zeros(200))
input1 = np.append(input1, np.ones(200))

# input 2 of the XOR gate
input2 = np.zeros(400)
input2 = np.append(input2, np.ones(400))

def create_structure(layers, neurons_per_layer,output_neurons, ip_weight, layer_weights, output_weight):
    ganglion = []
    for i in range(layers):
        layer = []
        if(i==0):
            for j in range(neurons_per_layer):
                layer.append(lif_neuron(ip_weight[:,j]))
            ganglion.append(layer)
        elif(i>0 and i<(layers-1)):
            for j in range(neurons_per_layer):
                layer.append(lif_neuron(layer_weights[:,j]))
            ganglion.append(layer)
        elif(i==(layers-1)):
            for j in range(output_neurons):
                layer.append(lif_neuron(output_weight))
            ganglion.append(layer)
        else:
            print("Invalid parameters for teh structure")
    return ganglion
            
def freq(tw,x,fmax): #provide a time window, flattened image (U_train[0]), and max frequency
    out=np.zeros((int(N),int(win[1]-win[0]))) # initialize output matrix
    
    for i in range(len((x))): # for each pixel in image
        spikes = round(((float((win[1]-win[0])))*fmax)*x[i]) # calculate the number of spikes
        spikeprob=spikes/float((win[1]-win[0])) # calculate the spike probability
        out[i]=np.random.choice(np.arange(0, 2), size=(win[1]-win[0]), replace=True, p=[1-spikeprob, spikeprob])
        # randomly populate output matrix with spikes according to spike probability
    return out

layers = 3
neurons_per_layer = 2
output_neurons = 1

win=[0,200] # input window in milliseconds
fm = 0.2 # max frequency 200Hz

# Verify the correct operation of frequency function
# r = number of spikes proportional to the greyscale magnitude of input pixels
W=freq(win,U_train[1],fm)
r=np.zeros((int(N)))
for e in range(N):
    r[e]=sum(W[e])

# ip_weight = 0.01 * np.random.randn(2,2)
# layer_weights = 0.01 * np.random.randn(2,2)
# output_weight= 0.01*np.random.randn(1,2)

ip_weight = np.array([[6.83553207, -60.52424094], [4.70213729, -60.66205472]])
layer_weights = np.array([[12.83553207, -50.52424094], [50.70213729, 50.66205472]])
output_weight = np.array([-9.5020065,10.44247489])


structure = create_structure(layers, neurons_per_layer, output_neurons, ip_weight, layer_weights, output_weight)

layer = 0
for i in range(len(input1)):
    for j in range(neurons_per_layer):
        structure[layer][j].fire([input1[i], input2[i]], i)

layer = 1
for i in range(len(input1)):
    for j in range(neurons_per_layer):
        structure[layer][j].fire([structure[0][0].spikes[i], structure[0][1].spikes[i]], i)

layer = 2
for i in range(len(input1)):
    for j in range(output_neurons):
        structure[layer][j].fire([structure[layer-1][0].spikes[i], structure[layer-1][1].spikes[i]], i)
        
fig, a = plt.subplots(2,2)
a[0][0].plot(np.arange(start=0, stop =800, step = 1), input1)
a[0][1].plot(np.arange(start=0, stop =800, step = 1), input2)
a[1][1].plot(np.arange(start=0, stop = 800, step =1), structure[2][0].spikes)