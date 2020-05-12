#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:37:11 2020

@author: alex
"""
import numpy as np
import copy

class TextSynthesizer():
    def __init__(self, data, m, eta, seq_length, sig):
        """ Initialize Class with parameters U, V, W, b, c contained in RNN
        
        Args:
            data               (dict):
                - char2ind     (dict): mapping of characters to indices
                - ind2char     (dict): mapping of characters to indices
                - num_chars     (int): number of characters in text
                - text_chars   (list): single text characters
                - text_data     (str): text which is used for learning
            m                   (int): dimensionality of hidden states
            eta               (float): learning rate
            seq_length          (int): length of the input sequence
            sig               (float): factor for initialization    
        """
        self.data = data
        self.m = m
        self.eta = eta
        self.seq_length = seq_length
        self.sig = sig
        
        # Pre-definitons and initialization
        self.H0 = np.zeros((self.m,1))
        self.K = self.data["num_chars"]               
        self.RNN, self.G = self.init_params()   
        
    def init_params(self):
        """ Initialize parameters U, V, W, b, c contained in RNN and the 
            G-matrices for parameters which is required for learning with 
            AdaGrad
            
        Returns:
            RNN (dict): contains the parameters of the network
            G   (dict): square of the gradients of the parameters (AdaGrad)
        """
        RNN = {"U": [], "V": [], "W": [], "b": [], "c": []}
        
        RNN["U"] = np.random.normal(0, self.sig, size = (self.m, self.K))
        RNN["W"] = np.random.normal(0, self.sig, size = (self.m, self.m))
        RNN["V"] = np.random.normal(0, self.sig, size = (self.K, self.m))
        RNN["b"] = np.zeros((self.m,1))
        RNN["c"] = np.zeros((self.K,1))    
        
        G = {}
        for k,_ in RNN.items():
            G[k] = 0
        return RNN, G
        
    def RNN_SGD(self):
        """ Training of the network with AdaGrad
            
        Returns:
            smooth_loss (list): losses for each update step
            step         (int): final number of update steps
        """
        # Settings
        n_epoch = 10
        step = 0
        # Pre-definitons
        l_text = len(self.data["text_data"]) # no. of characters in text
        checkpoints = np.array(range(0,l_text,10000))
        
        # Train network
        for epoch in range(n_epoch):
            Hprev = self.H0
            for e in range(0, l_text - self.seq_length -1, self.seq_length): 
                # Initialize inputs (training and validation data) for current
                # update step from text
                X, Y = self.init_inputs(e) 
                
                # Gradient computation and paramater update
                grad, Hprev, loss = self.computeGradients(X, Y, Hprev) 
                if step == 0 and epoch == 0:
                    smooth_loss = [loss]
                    self.compareGradientComputation(X, Y, self.H0, grad)
                for k,_ in self.RNN.items():
                    self.G[k] += grad[k] * grad[k]
                    self.RNN[k] -= self.eta / np.sqrt(self.G[k] + np.finfo(float).eps) * grad[k]
                    
                # Smooth loss
                smooth_loss.append(.999 * smooth_loss[step-1] + .001 * loss)
                # Print predicted text
                if any(step == checkpoints):
                    T,text = self.synthesizeText(X, Hprev, 1000)
                    print("\n" + "Iteration: " + str(step) + " of " + str(n_epoch*np.floor((l_text - self.seq_length -1)/self.seq_length)) + ", " + str(" {:.6f}".format(e/(l_text - self.seq_length -1)*100)) + ' %')                
                    print("Smooth loss: " + str(smooth_loss[step+1]))
                    print("\n" + text)
                # count update steps
                step += 1
        return smooth_loss, step
    
    def init_inputs(self, e):
        """ Initialize inputs (training and validation data) for current
            update step from text
            
        Args: 
            e (int): index of first character in text book sequence
        
        Returns:
            X (np.ndarray): one-hot endcoding of training data
            Y (np.ndarray): one-hot endcoding of validation data
        """
        X_chars = self.data["text_data"][e : e+self.seq_length]
        Y_chars = self.data["text_data"][e+1 : e+self.seq_length+1]
        X = np.zeros((self.K,self.seq_length))
        Y = np.zeros((self.K,self.seq_length))
        for n in range(self.seq_length):
            X[self.data["char2ind"][X_chars[n]], n] = 1
            Y[self.data["char2ind"][Y_chars[n]], n] = 1
        return X, Y
        
        
    def synthesizeText(self, X, Hprev, seq_length):
        """ Synthesizes text (str) with the learned parameters of the network
            and the first latter of the input data X
            
        Args: 
            X     (np.ndarray): one-hot endcoding of training data
            Hprev (np.ndarray): previously learned hidden state
            seq_length   (int): length of the input sequence
        Returns:
            T     (np.ndarray): one-hot endcoding matrix of synthesized text
            text         (str): synthesized text
        """
        # Define T vector: one-hot endcoding matrix of synthesized text
        T = np.zeros((self.K,seq_length+1))
        
        #
        ii = [None]*(seq_length+1)
        text = ""
        H = Hprev
        T[:,0] = X[:,0]
        ii[0] = int(np.where(X[:,0])[0])

        for t in range(seq_length):
            _, H, _, P = self.evaluate_classifier(T[:,t],self.RNN, H)
            ixs = np.where(np.cumsum(P) - np.random.rand(1) > 0)
            ii[t+1] = ixs[0][0]
            T[ii[t+1], t+1] = 1
            text = text + self.data["ind2char"][ii[t]]     
        return T, text
            
        
    def softmax(self, O):
        """ Compute the softmax
        Args:
            O (np.ndarray): output matrix (D, N)
        Returns a stable softmax matrix
        """
        P = np.exp(O - np.max(O, axis=0)) / \
                np.exp(O - np.max(O, axis=0)).sum(axis=0)
        return P
     
    def evaluate_classifier(self, X, RNN, H):    
        """ Scores learned classifier w.r.t. weight matrix and bias vector: log probability
        Args:
            X (np.ndarray): one-hot endcoding of training data
            RNN     (dict): contains the parameters of the network
            H (np.ndarray): hidden state of previous time-step 
        Returns:
            A (np.ndarray):
            H (np.ndarray): hidden state of this time-step 
            O (np.ndarray):
            H (np.ndarray): log probabilities for next letter
            
        """
        X = np.resize(X, (X.shape[0],1))
        A = RNN["W"]@H + RNN["U"]@X + RNN["b"]
        H = np.tanh(A)
        O = RNN["V"]@H + RNN["c"]
        P = self.softmax(O)
        return A, H, O, P
    
    def computeGradients(self, X, Y, Hprev):
        """ Compute the gradients of the parameter U, V, W, b, c analytically
            Forward + backward pass
        Args:
            X     (np.ndarray): one-hot endcoding of training data
            Y     (np.ndarray): one-hot endcoding of validation data
            Hprev (np.ndarray): previously learned hidden state
        Returns:
            Hprev (np.ndarray): final hidden state of this update step 
            loss     (float64): cross-entropy loss at the current update step
            grad        (dict): contains the gradients of the parameters
        """
        loss = 0  
        RNN = self.RNN
        A, H, O, P = {}, {}, {}, {}
        
        # Get hidden state of previous update step
        H[-1] = np.copy(Hprev)
        # Forward path
        for t in range(self.seq_length):
            A[t], H[t], O[t], P[t] = self.evaluate_classifier(X[:,t], RNN, H[t-1])
            loss += -np.log(np.sum(np.reshape(Y[:,t], (Y[:,t].shape[0],1))*P[t]))
        # Backward path
        grad = {"U": np.zeros_like(self.RNN["U"]), "V": np.zeros_like(self.RNN["V"]), "W": np.zeros_like(self.RNN["W"]),\
                "b": np.zeros_like(self.RNN["b"]), "c": np.zeros_like(self.RNN["c"]),\
                "A": np.zeros_like(A[0]), "H": np.zeros_like(H[0]), "O": np.zeros_like(O[0]),\
                "Anext": np.zeros_like(A[0])}
        for t in reversed(range(self.seq_length)):
            grad["O"] = -(np.reshape(Y[:,t], (Y[:,t].shape[0],1)) - P[t])
            grad["H"] = self.RNN["V"].T@grad["O"] + self.RNN["W"].T@grad["Anext"]
            grad["A"] = grad["H"]*(1-np.square(np.tanh(A[t])))
            grad["Anext"] = grad["A"]
            
            grad["U"] += grad["A"]@np.transpose(np.reshape(X[:,t], (X[:,t].shape[0],1)))
            grad["V"] += grad["O"]@np.transpose(np.reshape(H[t], (H[t].shape[0],1)))
            grad["W"] += grad["A"]@np.transpose(np.reshape(H[t-1], (H[t-1].shape[0],1)))
            
            grad["b"] += grad["A"]
            grad["c"] += grad["O"]
        
        # Clip gradients, min: -5, max: 5
        for k in grad:
            grad[k] = np.clip(grad[k], -5, 5)
        Hprev = H[self.seq_length-1]
        return grad, Hprev, loss
    
    def computeGradientsNum(self, X, Y, Hprev, n, h = 1e-5):
        """ Compute the gradients of the parameter U, V, W, b, c numerically
            (central differencing scheme w.r.t. loss)
        Args:
            X     (np.ndarray): one-hot endcoding of training data
            Y     (np.ndarray): one-hot endcoding of validation data
            Hprev (np.ndarray): previously learned hidden state
        Returns:
            grad_num    (dict): contains the numerically calculated gradients of the parameters
        """
        RNN = {}
        grad_num = {}
        for k,_ in self.RNN.items():
            grad_num.update({k: np.zeros_like(self.RNN[k])})
            RNN.update({k: self.RNN[k]})
            
            # n = np.size(RNN[k])
        for k,_ in self.RNN.items():
            for i in range(n):
                RNN_try = copy.deepcopy(RNN)
                RNN_try[k].flat[i] = RNN[k].flat[i] - h
                l1 = self.computeLoss(RNN_try, X, Y, Hprev)
                RNN_try[k].flat[i] = RNN[k].flat[i] + h
                l2 = self.computeLoss(RNN_try, X, Y, Hprev)
                grad_num[k].flat[i] = (l2 - l1) / (2*h)
        return grad_num

    
    def computeLoss(self, RNN, X, Y, Hprev):
        """ Calculate the forward pass for the calculation of the cross-entropy
            loss which is needed for the numerical gradient computation
        
        Args:
            RNN         (dict): contains the parameters of the network 
                                ("disturbed" by numerical step)
            X     (np.ndarray): one-hot endcoding of training data
            Y     (np.ndarray): one-hot endcoding of validation data
            Hprev (np.ndarray): previously learned hidden state
        Returns:
            loss     (float64): cross-entropy loss
        """
        loss = 0
        H = [None]*self.seq_length   
        P = [None]*self.seq_length   
        H[-1] = np.copy(Hprev)
        # Forward path
        for t in range(self.seq_length):
            _, H[t], _, P[t] = self.evaluate_classifier(X[:,t], RNN, H[t-1])
            loss += -np.log(np.sum(np.reshape(Y[:,t], (Y[:,t].shape[0],1))*P[t]))
        return loss
    
    
    def compareGradientComputation(self, X, Y, Hprev, grad, n = 20):
        """ Call the numerical gradient calculation and determine the errors 
            between analytical and numerical computation. print errors.
            
        Args:
            X     (np.ndarray): one-hot endcoding of training data
            Y     (np.ndarray): one-hot endcoding of validation data
            Hprev (np.ndarray): previously learned hidden state
            grad        (dict): contains the analytically computed gradients 
                                of the parameters            
        """
        grad_num = self.computeGradientsNum(X, Y, Hprev, n)
        grad_error_rel = {}
        grad_error_rel_max = {}
        for k,_ in grad_num.items():
            if k != "U":
                (grad[k].flat[:n] - grad_num[k].flat[:n]) / grad_num[k].flat[:n]
                grad_error_rel.update({k: (grad[k].flat[:n] - grad_num[k].flat[:n]) / grad_num[k].flat[:n]})
                grad_error_rel_max.update({k: np.max(grad_error_rel[k])})
            else:
                idx = grad[k].flat[:n] != 0
                (grad[k].flat[idx] - grad_num[k].flat[idx]) / grad_num[k].flat[idx]
                grad_error_rel.update({k: (grad[k].flat[idx] - grad_num[k].flat[idx]) / grad_num[k].flat[idx]})
                grad_error_rel_max.update({k: np.max(grad_error_rel[k])})

            print("Maximum relative error of " + k + ": " + str("{:.6f}".format(grad_error_rel_max[k]*100)) + ' %')