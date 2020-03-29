#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:07:15 2020

@author: alex
"""
import numpy as np

class MiniBatchGD():
    def __init__(self, X, Y, y, GDparams, W, b):
        self.X = X
        self.Y = Y
        self.y = y
        self.GDparams = GDparams
        self.W = W
        self.b = b
        self.lamda = GDparams[3]
        self.eta = GDparams[2]
        
    """ Training functions """

    def ScoreClassifier(self):
        self.S = (self.W).dot(self.X) + np.transpose(self.b * np.ones(((self.X).shape[1],1)))
        return self.S
    
    def Softmax(self):
        self.ScoreClassifier()
        self.P = np.exp(self.S) / np.sum(np.exp(self.S), axis=0)
        return self.P
    
    def ComputeGradients(self): 
        self.Softmax()
        G = -(self.Y - self.P)
        self.gradL_W = (1/(self.X).shape[1]) * G.dot(np.transpose(self.X))
        self.gradL_b = (1/(self.X).shape[1]) * G.dot(np.ones((self.X).shape[1]))
        self.grad_W = self.gradL_W + 2*self.lamda*self.W
        self.grad_b = self.gradL_b
        return [self.grad_W, self.grad_b]
    
    def ComputeParameters(self):
        self.ComputeGradients()
        self.W = self.W - self.eta * self.grad_W
        self.b = self.b - self.eta * self.grad_b 
        return [self.W, self.b]
    
    """Evaluate Training"""

    def ComputeCost(self): 
        P_y = np.dot(np.transpose(self.Y), self.P)        # matrix multiplication: Y(tranposed)*P
        p_y = np.diag(P_y)                      # extract diagonal elements
        l_cross = -np.log(p_y)
        self.l_cross_sum = np.sum(l_cross)           # cross-entropy loss
        self.L =  1/(self.Y).shape[1] * self.l_cross_sum                                    # total loss
        r = self.lamda*np.power(np.linalg.norm(self.W),2) # regularization term
        self.J = self.L + r
        return [self.L, self.J]

    def ComputeAccuracy(self):
        self.Softmax()
        k_star = np.argmax(self.P, axis=0)               # label with highest probability
        self.acc = len(np.argwhere(k_star - (self.y) ==0)) / (self.y).shape[1]
        return self.acc




