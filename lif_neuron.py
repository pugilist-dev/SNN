#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 22:57:16 2020

@author: rajiv
"""

import numpy as np
from scipy.integrate import quad


class lif_neuron(object):
    def __init__(self, weights,thresh = 0.5, rm = 1, cm = 10):
        self.mem_volt = np.array([])
        self.spikes = np.array([])
        self.tik = 0
        self.rm = rm
        self.cm = cm
        self.tau = self.rm * self.cm
        self.ref = 4
        self.thresh = thresh
        self.volt_spike = 1
        self.pre_sntc_wght = np.array(weights)
        self.v_rest = -40
        self.cache ={
            "event":[],
            "event_time":[]}
        
    def eta(self, t):
        """
        Evaluate the eta function
        Parameters
        t: current time
        tik: time since last spike
        """
        return (self.v_rest * np.exp((-1)*(t-self.tik)/self.tau))
    
    def integrand(self, time, stimulus):
        return np.exp((-1)*time/self.tau)*stimulus
        
    def t_minus_tk(self, time, tk):
        """
        Evaluate t - tik or t-tjk
        Parameters: 
        t: Current time
        tk: tik(Post synaptic response time) or tjk (presynaptic response time) 
        """
        return time - tk
        
        
    def fire(self, stimulus, time):
        spk_shape = self.eta(time)
        epsilon_matrix = np.array([])
        t_minus_tik = self.t_minus_tk(time, self.tik)
        for i in range(len(stimulus)):
            I = quad(self.integrand, 0, t_minus_tik, args=(stimulus[i]))
            epsilon = (1/self.cm)** np.array(I)
            epsilon_matrix = np.append(epsilon_matrix, epsilon)
        spike = spk_shape + np.sum(self.pre_sntc_wght)*np.sum(epsilon_matrix)
        #self.mem_volt = np.append(self.mem_volt, spike)
        if(spike > -0.5):
            self.tik = time
            self.mem_volt = np.append(self.mem_volt, spike)
            self.spikes = np.append(self.spikes, 1)
        else:
            self.mem_volt = np.append(self.mem_volt, spike)
            self.spikes = np.append(self.spikes, 0)