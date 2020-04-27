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
    
    def __init__(self, data, label_names, layers, init_type, batch_norm, W=None, b=None):
        """ Initialize Class with parameters W and b
        
        Args:
            data        (dict):
                - training, validation and testing:
                    - image data matrix                 X
                    - one-hot-encoded labels matrix     Y
                    - labels vector
            label_names (dict): label names (strings)
            layers      (dict): define layers with shapes and activations
            init_type    (str): type of initialization (Xavier, He, sigma)
            batch_norm  (bool): batch normalization (y/n)
            W     (np.ndarray): weight matrix
            b     (np.ndarray): bias matrix       
        """
        
        for k, v in data.items():
            setattr(self, k, v)
                        
        self.label_names = label_names
        self.k_num = len(layers)
        self.init_type = init_type
        self.batch_norm = batch_norm
            
        """ Initialize W and b with mean = 0, std = 0.01, size=(dim0: # of labels, dim1: pixel data)"""
        
        self.W = []
        self.b = []
        self.beta = []
        self.gamma = []
        self.activations = []
        self.mu_movavg = []
        self.var_movavg = []
        self.alpha = 0.9
        
        # Create initialization arrays for parameters W, b(), beta, gamma)
        for l in layers.values():
            for k, v in l.items():          # k: field identifier, v: field value
                if k == 'shape':
                    W, b, beta, gamma, mu_movavg, var_movavg = self.init_params(v, self.init_type)   # v: shape of layer l
                    self.W.append(W)
                    self.b.append(b)
                    self.beta.append(beta)
                    self.gamma.append(gamma)
                    self.mu_movavg.append(mu_movavg)
                    self.var_movavg.append(var_movavg)
                elif k == 'activation':
                    self.activations.append(v)  # v: activation function (str)
                    
        # Create dictionary with gradients for each parameter
        if self.batch_norm:
            self.params = {'W': self.W, 'b': self.b, 'beta': self.beta, 'gamma': self.gamma}
        else:
            self.params = {'W': self.W, 'b': self.b}
                    
    """ Define parameter initialization """       
            
    def init_params(self, shape, init_type):
        if init_type == 'He':
            W = (np.random.normal(0, 1/np.sqrt(shape[1]), size = (shape[0], shape[1])))
        elif init_type == 'Xavier':
            W = (np.random.normal(0, 2/np.sqrt(shape[1]), size = (shape[0], shape[1])))
        elif init_type == 'sigma':
            W = (np.random.normal(0, 1e-4, size = (shape[0], shape[1])))
        else:
            print('error: no proper weight initialization method chosen')            
        b = np.zeros((shape[0], 1))
        beta = np.zeros((shape[0], 1))
        gamma = np.ones((shape[0], 1))
        mu_movavg = np.zeros((shape[0], 1))
        var_movavg = np.zeros((shape[0], 1))
        return W, b, beta, gamma, mu_movavg, var_movavg
            
    """ Training functions """
    
    def mini_batch_gd(self, X, Y, y, lamda, eta, n_batch, n_epoch, n_s,
                      eval_performance = False, eta_is_cyclic = False):
        
        """ Training of the network w.r.t. the optimization of the parameters
            W and b with mini-batch gradient descent
            
        Args:
            X          (np.ndarray):    data matrix (D, N)
            Y          (np.ndarray):    one-hot-encoding labels matrix (C, N)
            y               (uint8):    label allocation vector (1, N)
            lamda           (float):    regularization term
            n_batch           (int):    number of batches
            eta             (float):    learning rate
            n_epoch           (int):    number of training epochs
            n_s               (int):    number of update steps
            eval_performance (bool):    sets whether to calculate total loss and cost   
            eta_is_cyclic    (bool):    sets whether a cyclical learning rate is applied
        Returns:
            self.W     (np.ndarray):    output weight matrix
            acc_train  (np.ndarray):    the accuracy on the training set for each epoch
            acc_val    (np.ndarray):    the accuracy on the validation set for each epoch
            acc_test   (np.ndarray):    the accuracy on the testing set for each epoch
            L_train    (np.ndarray):    the cross-entropy loss on the testing set for each epoch
            J_train    (np.ndarray):    the cost on the testing set for each epoch
            L_val      (np.ndarray):    the cross-entropy loss on the validation set for each epoch
            J_val      (np.ndarray):    the cost on the validation set for each epoch
            eta_s      (np.ndarray):    the learning rate for each update-step            
        """
                
        l_batch = int(X.shape[1] / n_batch) # length of batches
        
        L_train = np.zeros(n_epoch)
        J_train = np.zeros(n_epoch)
        L_val = np.zeros(n_epoch)
        J_val = np.zeros(n_epoch)
        
        acc_train = np.zeros(n_epoch)
        acc_val = np.zeros(n_epoch)
        acc_test = np.zeros(n_epoch)
        
        eta_s = np.zeros(n_epoch*n_batch)
        
        l = 0       # control variable
        
        
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
                
                # current update-step               
                t = j*n_batch + i + 1 
                # compute learning rate for current update-step t:
                eta_s[t-1], l = self.cyclic_learningrate(eta, n_s, l, t)
                
                self.batch_is = i # mini-batch counter
                grad = self.compute_gradients(X_batch, Y_batch, lamda)
                
                # compute delta of parameters for step
                delta_W = np.multiply(-eta_s[t-1], grad['W'])
                delta_b = np.multiply(-eta_s[t-1], grad['b'])
                delta_beta = np.multiply(-eta_s[t-1], grad['beta'])
                delta_gamma = np.multiply(-eta_s[t-1], grad['gamma'])
                for i in range(len(delta_b)):
                    delta_b[i] = delta_b[i].reshape((-1,1)) # reshape from (dim0, ,) to (dim0, 1)
                    if self.batch_norm:
                        delta_beta[i] = delta_beta[i].reshape((-1,1)) # reshape from (dim0, ,) to (dim0, 1)
                        delta_gamma[i] = delta_gamma[i].reshape((-1,1)) # reshape from (dim0, ,) to (dim0, 1)
                    
                
                self.W = list(np.add(self.W, delta_W))
                self.b = list(np.add(self.b, delta_b))  
                if self.batch_norm:
                    self.beta = list(np.add(self.beta, delta_beta))
                    self.gamma = list(np.add(self.gamma, delta_gamma))
            
            acc_train[j] = self.compute_accuracy(X, y, is_training = False)
            acc_val[j] = self.compute_accuracy(self.X_val, self.y_val, is_training = False)
            acc_test[j] = self.compute_accuracy(self.X_test, self.y_test, is_training = False)
                
            if eval_performance:
                [L_train[j], J_train[j]] = self.compute_cost(X, Y, lamda, is_training = False)
                [L_val[j], J_val[j]] = self.compute_cost(self.X_val, self.Y_val, lamda, is_training = False)
        
        return [self.W, acc_train, acc_val, acc_test, L_train, J_train, L_val, J_val, eta_s]
    
    """ Forward Path
    Fns
        relu()
        softmax()
        evaluate_classifier()        
    """      
    def relu(self, S1):
        H = np.maximum(S1, 0)
        return H
    
    def softmax(self, S):
        """ Compute the softmax
        Args:
            S (np.ndarray): score matrix (D, N)
        Returns a stable softmax matrix
        """
        P = np.exp(S - np.max(S, axis=0)) / \
                np.exp(S - np.max(S, axis=0)).sum(axis=0)
        return P
     
    def evaluate_classifier(self, X, is_training):    
        """ Scores learned classifier w.r.t. weight matrix and bias vector: log probability
        Args:
            X         (np.ndarray): data batch matrix (D, N)
            is_training     (bool): define whether evaluation happens during test time
        Returns
            P     (np.ndarray): log probabilities of classification
            H           (list): intermediary activation functions of layers
            S           (list): score matrices of layers
            Shat        (list): batch normalized score matrices of layers
            means       (list): mean value per dimension of layers
            variances   (list): variances per dimension of layer
        """

        S = []
        Shat = []
        H = [np.copy(X)]
        means = []
        variances = []
        for l in range(0,self.k_num):
            # activation function applied on score  
            s = self.W[l]@H[l] + self.b[l]
            S.append(s)
            if self.activations[l] == 'relu':
                if self.batch_norm:
                    # calc mean along all values of s and variance along all dimensions of s
                    mu = np.mean(s, axis = 1, keepdims = True)
                    means.append(mu)
                    var = np.var(s, axis = 1, keepdims = True)
                    variances.append(var)
                    if is_training:
                        # upgrade moving averages with mu of current update step/mini-batch
                        if self.batch_is == 0:
                            self.mu_movavg[l] = mu
                            self.var_movavg[l] = var
                        else:
                            self.mu_movavg[l] = self.alpha * self.mu_movavg[l] + (1-self.alpha) * mu
                            self.var_movavg[l] = self.alpha * self.var_movavg[l] + (1-self.alpha) * var           
                        # calc shat: shat = BatchNorm()
                        shat = (s - mu) / np.sqrt(var + np.finfo(np.float64).eps)
                    else:
                        shat = (s - self.mu_movavg[l]) / np.sqrt(self.var_movavg[l] + np.finfo(np.float64).eps)
                    Shat.append(shat)
                    # calc stilde: stilde = np.multiply(self.gamma[l], shat) + self.beta[l]
                    stilde = np.multiply(self.gamma[l], shat) + self.beta[l]
                    # apply relu to stilde: S = self.relu(stilde)
                    H.append(self.relu(stilde))
                else:
                    H.append(self.relu(s)) # intermediary activation function
            elif self.activations[l] == 'softmax':
                P = self.softmax(s)        # probability
        return [P, H, S, Shat, means, variances]
        
    """ Backward Path
    Fns
        do_backpassnorm()
        compute_gradients()
        compute_gradients_num()
    """   
    
    def do_backpassnorm(self, G, S, mu, var, n_b):
        """ Applies back pass for batch normalization
        Args:
            G           (np.ndarray): 
            S           (np.ndarray): score matrix of current layer
            means       (np.ndarray): mean value per dimension of current layers
            variances   (np.ndarray): variances per dimension of curren layer
            n_b                (int): dimensionality of data batch matrix X (no. of images)
        Returns
            G           (np.ndarray): 
        """
        n_g = G.shape[1]
        # Apply batch normalization for the back pass
        sigma1 = np.power(var + np.finfo(np.float64).eps, -0.5)
        sigma2 = np.power(var + np.finfo(np.float64).eps, -1.5)
        G1 = np.multiply(G, sigma1)
        G2 = np.multiply(G, sigma2)
        D = S - mu       
        c = np.sum(np.multiply(G2, D), axis=1, keepdims=True)         
        G = G1\
            - (1/n_g) * np.sum(G1, axis=1, keepdims=True)\
            - (1/n_g) * np.multiply(D, c)
        return G
    
    def compute_gradL(self, G, H, S, Shat, means, variances, n_b):
        """ Analytically computes the gradients of the loss w.r.t. the weight and bias parameters
        Args:          
            G     (np.ndarray): 
            H           (list): intermediary activation functions of layers               
            S           (list): score matrices of layers
            Shat        (list): batch normalized score matrices of layers
            means       (list): mean value per dimension of layers
            variances   (list): variances per dimension of layer
            n_b          (int): dimensionality of data batch matrix X (no. of images)
         Returns:
            gradL       (dict): for each layer - the gradients of the loss w.r.t. W, b, (beta, gamma) (np.ndarray)
            G     (np.ndarray): 
        """
        
        # Pre-definitions
        if self.batch_norm:
            gradL = {'W': [], 'b': [], 'beta': [], 'gamma': []}
        else:
            gradL = {'W': [], 'b': []}
        for l in range(self.k_num):
            for key in self.params:
                gradL[key].append([])
        # If batch normalization is applied        
        if self.batch_norm:       
            
            # Propagate gradient through loss and softmax operation (last layer)
            gradL['beta'][self.k_num-1] = np.zeros((self.params['beta'][self.k_num-1].shape[0],))
            gradL['gamma'][self.k_num-1] = np.zeros((self.params['gamma'][self.k_num-1].shape[0],))
            gradL['W'][self.k_num-1] = (1/n_b) * G@np.transpose(H[self.k_num-1])
            gradL['b'][self.k_num-1] = (1/n_b) * G@np.ones(n_b)
            G = np.transpose(self.W[self.k_num-1])@G            
            G = np.multiply(G, H[self.k_num-1] > 0)

            # Propagate from layer k-1(k_num-2) to layer 1(0)
            for l in range(self.k_num-2, -1, -1):   
                
                # Calculate gradient of losses for beta and gamma
                gradL['beta'][l] = (1/n_b) * G@np.ones(n_b)
                gradL['gamma'][l] = (1/n_b) * np.multiply(G, Shat[l])@np.ones(n_b)
                
                # Calulcate new G (gradient propagation through scale and shift)
                G = np.multiply(G, self.gamma[l]@np.ones((1,n_b)))
                
#                # Apply batch normalization for the back pass
                G = self.do_backpassnorm(G, S[l], means[l], variances[l], n_b)
                
                # Calculate gradient of losses for W and b
                gradL['W'][l] = (1/n_b) * G@np.transpose(H[l])
                gradL['b'][l] = (1/n_b) * G@np.ones(n_b)
                
                # Propagate until next layer
                if l > 0:
                    G = np.transpose(self.W[l])@G
                    G = np.multiply(G, H[l] > 0)                
        
        # If no batch normalization is applied    
        else:
            for l in range(self.k_num-1, -1, -1):
                gradL['W'][l] = (1/n_b) * G@np.transpose(H[l])
                gradL['b'][l] = (1/n_b) * G@np.ones(n_b)
                G = np.transpose(self.W[l])@G
                G = np.multiply(G, H[l] > 0)
        return [gradL, G]
    
    def compute_gradients(self, X_batch, Y_batch, lamda, is_training = True): 
        """ Analytically computes the gradients of the weight and bias parameters
        Args:
            X_batch (np.ndarray): data batch matrix (D, N)
            Y_batch (np.ndarray): one-hot-encoding labels batch vector (C, N)
            lamda        (float): regularization term
        Returns:
            grad          (dict): for each layer - the gradients of the cost w.r.t. W, b, (beta, gamma) (np.ndarray)
        """
        n_b = X_batch.shape[1] # number of images in dataset

        # Forward path
        [P, H, S, Shat, means, variances] = self.evaluate_classifier(X_batch, is_training = is_training)
        
        # Backward path
        G = -(Y_batch - P)
    
        grad = {'W': [], 'b': [], 'beta': [], 'gamma': []}

        [gradL, G] = self.compute_gradL(G, H, S, Shat, means, variances, n_b)  # gradients for layers 2 to k    
        for l in range(0,self.k_num):
            grad['W'].append(gradL['W'][l] + 2*lamda*self.W[l])
            grad['b'].append(gradL['b'][l])
            if self.batch_norm:
                grad['beta'].append(gradL['beta'][l])
                grad['gamma'].append(gradL['gamma'][l])
        return grad
    
    def compute_gradients_num(self, X_batch, Y_batch, lamda, h=1e-7):
        """Numerically computes the gradients of the weight and bias parameters
        Args:
            X_batch (np.ndarray): data batch matrix (D, N)
            Y_batch (np.ndarray): one-hot-encoding labels batch vector (C, N)
            lamda        (float): regularization term
            h            (float): step-size
        Returns:
            grad_num      (dict): for each layer - the numerically calculatedgradients of the loss w.r.t. W, b, (beta, gamma) (np.ndarray)
        """
        
        grad_num = {'W': [], 'b': [], 'beta': [], 'gamma': []}         
        
        for k in range(len(self.b)):
            for key in self.params:
                grad_num[key].append(np.zeros(self.params[key][k].shape))
                for i in range(len(self.params[key][k].flatten())):
                    param_try = self.params[key][k].flat[i]
                    self.params[key][k].flat[i] = param_try + h
                    c2, _ = self.compute_cost(X_batch, Y_batch, lamda)
                    self.params[key][k].flat[i] = param_try - h
                    c3, _ = self.compute_cost(X_batch, Y_batch, lamda)
                    grad_num[key][k].flat[i] = (c2-c3) / (2*h)
                if key in ('b', 'beta', 'gamma'):
                    grad_num[key][k] = np.squeeze(grad_num[key][k],axis=1)
        return grad_num
    
    """ Evaluate Training
    Fns
        compute_cost()
        compute_accuracy()
    """    

    def compute_cost(self, X, Y, lamda, is_training = True): 
        """ Computes the cost of the classifier using the cross-entropy loss
        Args:
            X (np.ndarray): data matrix (D, N)
            Y (np.ndarray): one-hot encoding labels matrix (C, N)
            lamda  (float): regularization term
        Returns:
            L (float): the total cross-entropy loss
            J (float): the cost function of the cross-entropy loss
        """
        W_norm = []
        n_b = X.shape[1] # number of images in dataset
        
        [P, _, _, _, _, _] = self.evaluate_classifier(X, is_training = is_training)
        l_cross_sum = np.sum(-Y*np.log(P))
        L =  1/n_b * l_cross_sum                        # total loss
        for i in range(len(self.W)):
            W_norm.append(np.power(np.linalg.norm(self.W[i]),2))
        r = lamda*np.sum(W_norm)                        # regularization term
        J = L + r                                       # cost function of cross-entropy loss
        return [L, J]

    def compute_accuracy(self, X, y, is_training = True):
        """ Computes the accuracy of the classifier
        Args:
            X (np.ndarray): data matrix (D, N)
            y (np.ndarray): labels vector (N)
        Returns:
            acc    (float): the accuracy of the classification on the given data matrix X
        """
       
        [P, _, _, _, _, _] = self.evaluate_classifier(X, is_training = is_training)       
        k_star = np.argmax(P, axis=0)               # label with highest probability
        acc = len(np.argwhere(k_star - y == 0)) / y.shape[1]
        return acc
    
    def cyclic_learningrate(self, eta, n_s, l, t):
        """ Computes the learning rate for the current learning cycle
        Args:
            eta         (list): contains min/max learning rates
            n_s        (float): 2n_s - no. of update-steps per learning cycle
            l            (int): current learning cycle
            t            (int): running variable: update-step     
        Returns:
            eta_s (np.ndarray): learning rate for current update-step, stored with previous learning rates
            l            (int): current learning cycle
        """
        if t-2*(l+1)*n_s == 1:
            l = l + 1
        if t >= 2*l*n_s and t <= (2*l+1)*n_s:
            eta_s = eta[0] + (eta[1] - eta[0])/n_s * (t - 1 - 2*l*n_s)
        elif t >= (2*l+1)*n_s and t <= 2*(l+1)*n_s:
            eta_s = eta[1] - (eta[1] - eta[0])/n_s * (t - 1 - (2*l+1)*n_s)
        return eta_s, l