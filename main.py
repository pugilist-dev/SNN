#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 22:35:02 2020

@author: rajiv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import heapq
import math
from scipy.io import loadmat


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

layers = 10
neurons_per_layer = 10


class lif_neuron(object):
    def __init__(self, weights,thresh = 0.5, rm = 1, cm = 10):
        self.mem_volt = np.array([0])
        self.spikes = np.array([0])
        self.t = 0
        self.rm = rm
        self.cm = cm
        self.tau = self.rm * self.cm
        self.ref = 4
        self.thresh = thresh
        self.volt_spike = 1
        self.pre_sntc_wght = np.array(weights)
        self.v_rest = -40
        self.w = -5.0
        self.cache ={
            "event":[],
            "event_time":[]}
    
    def eta(self, t):
        """
        Evaluate the eta function
        parameters
        t = time(t-tik)
        ets(t) = s_reset * exp(t/tau)
        returns of a floating type 
        """
        return (self.v_rest * np.exp(t//self.tau))
    
    def epsilon(self, stimulus, t):
        """
        Evaluates the post synaptic potential
        parameters:
            stimulus: the spike train that is the input of the neuron
            t: t-tik
        """
        return stimulus*(-1*self.tau)*(np.exp(-t//self.tau))
    
    def fire(self, stimulus, spike_time, structure):
        """
        Evaluate the spike at all the time instances
        parameters
        stimulus: The spike train stimulus
        spike_time: The time instance at where there are spikes
        structure: entire network structure
        """
        simulation_window = len(stimulus)                 # Total time of the simulation window
        mem_volt = np.zeros(simulation_window)                 # Membrane voltage along the simulation window
        #spike = np.zeros(simulation_window)
        t_minus_tk = []
        j = 0                                       # output of the neuron
        for i in range(simulation_window):
            last_spike = spike_time[0][j]
            if((j+1 < len(spike_time[0])) and (i>last_spike) and (i<spike_time[0][j+1])):
                if((i - last_spike)> 0):
                    t_minus_tk.append(i- last_spike)
                else:
                    t_minus_tk.append(0)
            elif((j+1 < len(spike_time[0])) and (i==spike_time[0][j+1])):
                j+=1
                t_minus_tk.append(0)
            elif(i<last_spike):
                t_minus_tk.append(0)
            else:
                t_minus_tk.append(0)
        
        spk_shape = self.eta(np.asarray(t_minus_tk))
        epsilon = self.epsilon(stimulus, np.asarray(t_minus_tk))
        spike = spk_shape + epsilon*self.w
        filt = spike > 0.5
        spike[filt] = 1
        filt = spike < 0
        spike[filt] =0
        return spike


def encoding(data, time_window=0):
    data = data[:time_window]
    spikes = data.nonzero()
    return spikes

def input_weights(weights_dim):
    rows = weights_dim[0]
    columns = weights_dim[1]
    weights = 0.01 * np.random.randn(rows, columns)
    return weights

spike_time = np.array(encoding(U_train[0], 784))
weights_dim = (len(U_train[0]), neurons_per_layer)

ip_weight = input_weights(weights_dim)
   
def create_neurons(layers, neurons_per_layer):
    ganglion = []
    for i in range(layers):
        layer = []
        for j in range(neurons_per_layer):
            layer.append(lif_neuron(ip_weight[:,j]))
        ganglion.append(layer)
    return ganglion

structure = create_neurons(layers, neurons_per_layer)


layer = 0
stimulus = U_train[0]
spikes = structure[layer][0].fire(stimulus, spike_time, structure)
fig, a = plt.subplots(2,2) 
a[0][0].plot(np.arange(start=0, stop=784, step=1), spikes)
a[0][1].plot(np.arange(start=0, stop=784, step=1), U_train[0])