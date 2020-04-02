#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:07:15 2020

@author: Alexander Polidar
"""
import numpy as np

class ClassifierMiniBatchGD():
    """ Mini-batch gradient descent classifier
    
    Fns
        mini_batch_gd: minit-batch gradient descent
            # Forward path
                score_classifier:       Scores learned classifier w.r.t. weight matrix and bias vector: log probability        
                softmax:                Calculates prediction probabilities of learned classifier from log probabilities
            # Backward path
                compute_gradients       Computes gradients of W and b with respect to loss analytically
                compute_gradients_num   Computes gradients of W and b with respect to loss numerically (central differencing scheme)
            # Evaluation
                compute_cost            Computes cost function of the classifier with learned parameters using the cross-entropy loss
                compute_accuracy        Computes the accuracy of the classifier with learned parameters
    """
    
    def __init__(self, data, label_names, W=None, b=None):
        """ Initialize Class with parameters W and b
        
        Args:
            data (dict):
                - training, validation and testing:
                    - image data matrix                 X
                    - one-hot-encoded labels matrix     Y
                    - labels vector
            label_names (dict): label names (strings)
            W (np.ndarray): weight matrix
            b (np.ndarray): bias matrix
        
        """
        
        for k, v in data.items():
            setattr(self, k, v)
                        
        self.label_names = label_names
            
        """ Initialize W and b with mean = 0, std = 0.01, size=(dim0: # of labels, dim1: pixel data)"""
        
        self.W = W if W != None else np.random.normal(                          
                0, 0.01, (len(self.label_names), self.X_train.shape[0]))

        self.b = b if b != None else np.random.normal(
                0, 0.01, (len(self.label_names), 1))
                
    """ Training functions """
    
    def mini_batch_gd(self, X, Y, y, lamda, eta, n_batch, n_epoch,
                      eval_performance = False):
        
        """ Training of the network w.r.t. the optimization of the parameters
            W and b with mini-batch gradient descent
            
        Args:
            X    (np.ndarray): data matrix (D, N)
            Y    (np.ndarray): one-hot-encoding labels matrix (C, N)
            y         (uint8): label allocation vector (1, N)
            lamda     (float): regularization term
            n_batch     (int): number of batches
            eta       (float): learning rate
            n_epoch    (int): number of training epochs
            verbose    (bool): decides on textual output
            eval_performance (bool): decides whether to calculate total loss and cost            
        Returns:
            acc_train (float): the accuracy on the training set
            acc_val   (float): the accuracy on the validation set
            acc_test  (float): the accuracy on the testing set
        """
                
        l_batch = int(X.shape[1] / n_batch) # length of batches
        
        L_train = np.zeros(n_epoch)
        J_train = np.zeros(n_epoch)
        L_val = np.zeros(n_epoch)
        J_val = np.zeros(n_epoch)
        
        acc_train = np.zeros(n_epoch)
        acc_val = np.zeros(n_epoch)
        acc_test = np.zeros(n_epoch)
        
        # mini-batch iteration:
        # j-loop: epoch-loop
        # i-loop: batch-loop        
        for j in range(n_epoch): 
            # shuffle training data
            randperm = np.random.permutation(X.shape[1])    
            X_perm = X[:,randperm]
            Y_perm = Y[:,randperm]
            for i in range(n_batch):
                print('\r        Epoch ' + str(j+1) + ' - progress: ' + str(100*(i+1)/n_batch) + " %", flush=True, end='')
                i_start = i * l_batch
                i_end = (i+1) * l_batch
                idx = np.arange(i_start, i_end)
                
                X_batch = X_perm[:, idx]
                Y_batch = Y_perm[:, idx]   
                
                [self.grad_W, self.grad_b] = self.compute_gradients(X_batch, Y_batch, lamda)
                
                self.W -= eta * self.grad_W
                self.b -= eta * self.grad_b
                
            
            acc_train[j] = self.compute_accuracy(X, Y, y)
            acc_val[j] = self.compute_accuracy(self.X_val, self.Y_val, self.y_val)
            acc_test[j] = self.compute_accuracy(self.X_test, self.Y_test, self.y_test)
                
            if eval_performance:
                [L_train[j], J_train[j]] = self.compute_cost(X, Y, lamda)
                [L_val[j], J_val[j]] = self.compute_cost(self.X_val, self.Y_val, lamda)
        
        return [self.W, acc_train, acc_val, acc_test, L_train, J_train, L_val, J_val]
    
    """ Forward Path
    Fns
        score_classifier()
        softmax()
    """       
            
    def score_classifier(self, X):    
        """ Scores learned classifier w.r.t. weight matrix and bias vector: log probability
        Args:
            X (np.ndarray): data matrix (D, N)
        Returns log probabilities of classification
        """
        
        S = self.W@X + self.b
        return S
    
    def softmax(self, X):
        """ Compute the softmax
        Args:
            X (np.ndarray): data matrix (D, N)
        Returns a stable softmax matrix
        """
        
        S = self.score_classifier(X)
#        self.P = np.exp(self.S) / np.sum(np.exp(self.S), axis=0)
        P = np.exp(S - np.max(S, axis=0)) / \
                np.exp(S - np.max(S, axis=0)).sum(axis=0)
        return P
    
    """ Backward Path
    Fns
        compute_gradients()
        compute_gradients_num()
    """   
    
    def compute_gradients(self, X_batch, Y_batch, lamda): 
        """ Analytically computes the gradients of the weight and bias parameters
        Args:
            X_batch (np.ndarray): data batch matrix (D, N)
            Y_batch (np.ndarray): one-hot-encoding labels batch vector (C, N)
            lamda        (float): regularization term
        Returns:
            grad_W (np.ndarray): the gradient of the weight parameter
            grad_b (np.ndarray): the gradient of the bias parameter
        """
        P = self.softmax(X_batch)
        G = -(Y_batch - P)
        self.gradL_W = (1/(X_batch.shape[1])) * G@np.transpose(X_batch)
        self.gradL_b = (1/(X_batch.shape[1])) * G@np.ones(X_batch.shape[1])
        self.grad_W = self.gradL_W + 2*lamda*self.W
        self.grad_b = np.reshape(self.gradL_b, ((self.gradL_b).shape[0], 1))
        return [self.grad_W, self.grad_b]
    
    def compute_gradients_num(self, X_batch, Y_batch, lamda, h=1e-6):
        """Numerically computes the gradients of the weight and bias parameters
        Args:
            X_batch (np.ndarray): data batch matrix (D, N)
            Y_batch (np.ndarray): one-hot-encoding labels batch vector (C, N)
            lamda        (float): regularization term
            h            (float): step-size
        Returns:
            grad_W (np.ndarray): the gradient of the weight parameter
            grad_b (np.ndarray): the gradient of the bias parameter
        """
        grad_W = np.zeros(self.W.shape)
        grad_b = np.zeros(self.b.shape)

        b_try = np.copy(self.b)
        for i in range(len(self.b)):
            self.b = b_try
            self.b[i] = self.b[i] + h
            c2, _ = self.compute_cost(X_batch, Y_batch, lamda)
            self.b[i] = self.b[i] - 2*h
            c3, _ = self.compute_cost(X_batch, Y_batch, lamda)
            grad_b[i] = (c2-c3) / (2*h)

        W_try = np.copy(self.W)
        for i in np.ndindex(self.W.shape):
            self.W = W_try
            self.W[i] = self.W[i] + h
            c2, _ = self.compute_cost(X_batch, Y_batch, lamda)
            self.W[i] = self.W[i] - 2*h
            c3, _ = self.compute_cost(X_batch, Y_batch, lamda)
            grad_W[i] = (c2-c3) / (2*h)

        return grad_W, grad_b
    
    """ Evaluate Training
    Fns
        compute_cost()
        compute_accuracy()
    """    

    def compute_cost(self, X, Y, lamda): 
        """ Computes the cost of the classifier using the cross-entropy loss
        Args:
            X (np.ndarray): data matrix (D, N)
            Y (np.ndarray): one-hot encoding labels matrix (C, N)
            lamda  (float): regularization term
        Returns:
            L (float): the total cross-entropy loss
            J (float): the cost function of the cross-entropy loss
        """
        
        P = self.softmax(X)
        P_y = np.transpose(Y)@P                         # matrix multiplication: Y(tranposed)*P
        p_y = np.diag(P_y)                              # extract diagonal elements
        l_cross = -np.log(p_y)
        l_cross_sum = np.sum(l_cross)                   # cross-entropy loss
        L =  1/(Y.shape[1]) * l_cross_sum               # total loss
        r = lamda*np.power(np.linalg.norm(self.W),2)    # regularization term
        J = L + r                                       # cost function of cross-entropy loss
        return [L, J]

    def compute_accuracy(self, X, Y, y):
        """ Computes the accuracy of the classifier
        Args:
            X (np.ndarray): data matrix (D, N)
            y (np.ndarray): labels vector (N)
        Returns:
            acc (float): the accuracy of the classification on the given data matrix X
        """
        
        P = self.softmax(X)
        k_star = np.argmax(P, axis=0)               # label with highest probability
        acc = len(np.argwhere(k_star - y == 0)) / y.shape[1]
        return acc