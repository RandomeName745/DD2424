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
    
    def __init__(self, data, label_names, M, layers, init_type, batch_norm, W=None, b=None):
        """ Initialize Class with parameters W and b
        
        Args:
            data (dict):
                - training, validation and testing:
                    - image data matrix                 X
                    - one-hot-encoded labels matrix     Y
                    - labels vector
            label_names (dict): label names (strings)
            W     (np.ndarray): weight matrix
            b     (np.ndarray): bias matrix
            k_num        (int): number of layers
        
        """
        
        for k, v in data.items():
            setattr(self, k, v)
                        
        self.label_names = label_names
        self.M = M
        self.k_num = len(layers)
        self.init_type = init_type
        self.batch_norm = batch_norm
            
        """ Initialize W and b with mean = 0, std = 0.01, size=(dim0: # of labels, dim1: pixel data)"""
        
        self.W = []
        self.b = []
        self.beta = []
        self.gamma = []
        self.activations = []
        
        # Create initialization arrays for parameters W, b(), beta, gamma)
        for l in layers.values():
            for k, v in l.items():          # k: field identifier, v: field value
                if k == 'shape':
                    W, b, beta, gamma = self.init_params(v, self.init_type)   # v: shape of layer l
                    self.W.append(W)
                    self.b.append(b)
                    self.beta.append(beta)
                    self.gamma.append(gamma)
                elif k == 'activation':
                    self.activations.append(v)  # v: activation function (str)
                    
        # Create dictionary with gradients for each parameter
        if self.batch_norm:
            self.params = {'W': self.W, 'b': self.b, 'beta': self.beta, 'gamma': self.gamma}
        else:
            self.params = {'W': self.W, 'b': self.b}
                    
        
        # # Initialize first layer
        # self.W = W if W != None else [np.random.normal(0, 1/np.sqrt(self.X_train.shape[0]), (self.M, self.X_train.shape[0]))]   # W1 (M, D)
        # self.b = b if b != None else [np.zeros((self.M, 1))]                                                                    # b1 (M, 1)
        
        # # Initialize remaining layers
        # for k in range(self.k_num-1):
        #     self.W.append(np.random.normal(0, 1/np.sqrt(self.M), (len(self.label_names), self.M)))
        #     self.b.append(np.zeros((len(self.label_names), 1)))  
            
    """ Define parameter initialization """       
            
    def init_params(self, shape, init_type):
        if init_type == 'He':
            W = (np.random.normal(0, 1/np.sqrt(shape[1]), size = (shape[0], shape[1])))
        elif init_type == 'Xavier':
            W = (np.random.normal(0, 2/np.sqrt(shape[1]), size = (shape[0], shape[1])))
        else:
            print('error: no proper weight initialization method chosen')            
        b = np.zeros((shape[0], 1))
        beta = np.zeros((shape[0], 1))
        gamma = np.zeros((shape[0], 1))
        return W, b, beta, gamma
            
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
            verbose          (bool):    decides on textual output
            eval_performance (bool):    sets whether to calculate total loss and cost   
            eta_is_cyclic    (bool):    sets whether a cyclical learning rate is applied
        Returns:
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
                
                grad = self.compute_gradients(X_batch, Y_batch, lamda)
                
                # compute delta of parameters for step
                delta_W = np.multiply(-eta_s[t-1], grad['W'])
                delta_b = np.multiply(-eta_s[t-1], grad['b'])
                for i in range(len(delta_b)):
                    delta_b[i] = delta_b[i].reshape((-1,1)) # reshape from (dim0, ,) to (dim0, 1)
                    
                
                self.W = list(np.add(self.W, delta_W))
                self.b = list(np.add(self.b, delta_b))                
            
            acc_train[j] = self.compute_accuracy(X, y)
            acc_val[j] = self.compute_accuracy(self.X_val, self.y_val)
            acc_test[j] = self.compute_accuracy(self.X_test, self.y_test)
                
            if eval_performance:
                [L_train[j], J_train[j]] = self.compute_cost(X, Y, lamda)
                [L_val[j], J_val[j]] = self.compute_cost(self.X_val, self.Y_val, lamda)
        
        return [self.W, acc_train, acc_val, acc_test, L_train, J_train, L_val, J_val, eta_s]
    
    """ Forward Path
    Fns
        relu()
        score_classifier()
        softmax()
    """      
    def relu(self, S1):
        H = np.maximum(S1, 0)
        return H
    
    def softmax(self, S):
        """ Compute the softmax
        Args:
            X (np.ndarray): data matrix (D, N)
        Returns a stable softmax matrix
        """
        P = np.exp(S - np.max(S, axis=0)) / \
                np.exp(S - np.max(S, axis=0)).sum(axis=0)
        return P
     
    def evaluate_classifier(self, X):    
        """ Scores learned classifier w.r.t. weight matrix and bias vector: log probability
        Args:
            X (np.ndarray): data matrix (D, N)
        Returns log probabilities of classification
        """
        S = np.copy(X)
        H = [np.copy(X)]
        for l in range(0,self.k_num):
            if self.activations[l] == 'relu':
                
                # S.append(self.relu(self.W[l]@S[l] + self.b[l]))     # activation function
                # activation function applied on score  
                s = self.W[l]@S + self.b[l]
                if self.batch_norm:
                    # calc mean along all values of s and variance along all dimensions of s
                    mu = np.mean(s)
                    var = np.var(s, axis = 1)
                    # calc shat: shat = BatchNorm()
                    shat = (1/np.sqrt(np.diag(var + np.finfo(np.float64).eps)))@(s - mu)
                    # calc stilde: stilde = np.multiply(self.gamma[l], shat) + self.beta[l]
                    stilde = np.multiply(self.gamma[l], shat) + self.beta[l]
                    # apply relu to stile: S = self.relu(stilde)
                    S = self.relu(stilde)
                else:
                    S = self.relu(s)
                H.append(S) # intermediary activation function
                # Shat.append(np.diag(mu[l]))
            elif self.activations[l] == 'softmax':
                P = self.softmax(self.W[l]@S + self.b[l])        # probability
        return [P, H]
        
    """ Backward Path
    Fns
        compute_gradients()
        compute_gradients_num()
    """   
    
    def compute_gradL(self, G, H, n_b):
        """ Analytically computes the gradients of the loss w.r.t. the weight and bias parameters
        Args:          
            H       (np.ndarray): activation matrix H/data batch matrix X of previous layer (see lecture 5, slide 36, 2.)
            G       (np.ndarray): 
            n_b            (int): number of images in dataset
         Returns:
            gradL_W (np.ndarray): the gradient of the loss w.r.t. the weight parameter
            gradL_b (np.ndarray): the gradient of the loss w.r.t. the bias parameter
        """
        if self.batch_norm:
            gradL = {'W': [], 'b': [], 'beta': [], 'gamma': []}
        else:
            gradL = {'W': [], 'b': []}
        for l in range(self.k_num):
            for key in self.params:
                gradL[key].append([])
        
        for l in range(self.k_num-1, -1, -1):
            gradL['W'][l] = (1/n_b) * G@np.transpose(H[l])
            gradL['b'][l] = (1/n_b) * G@np.ones(n_b)
            G = np.transpose(self.W[l])@G
            G = np.multiply(G, H[l] > 0)
        return [gradL, G]
    
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
        n_b = X_batch.shape[1] # number of images in dataset
        # S = [X_batch]
        # P = []
        
        # Forward path
        # for l in range(0,self.k_num):
        #     [P, S] = self.evaluate_classifier(P, S, l)
        [P, H] = self.evaluate_classifier(X_batch)
        
        # Backward path
        G = -(Y_batch - P)
        
        # gradL_W = []
        # gradL_b = []
        # if self.batch_norm:
        #     gradL = {'W': [], 'b': [], 'beta': [], 'gamma': []}
        # else:
        #     gradL = {'W': [], 'b': []}
        # for l in range(self.k_num):
        #     for key in self.params:
        #         gradL[key].append([])
        grad = {'W': [], 'b': []}

        # for l in range(self.k_num-1, -1, -1):
        [gradL, G] = self.compute_gradL(G, H, n_b)  # gradients for layers 2 to k    
        for l in range(0,self.k_num):
            grad['W'].append(gradL['W'][l] + 2*lamda*self.W[l])
            grad['b'].append(gradL['b'][l])
        # grad['W'].reverse()
        # grad['b'].reverse()
        return grad
    
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
        
        grad_num = {'W': [], 'b': []}
        
        b_try = []
        W_try = []
        
        for k in range(len(self.b)):
            grad_num['W'].append(np.zeros(self.W[k].shape))
            grad_num['b'].append(np.zeros(self.b[k].shape))
            W_try.append(self.W[k])
            b_try.append(self.b[k])
            
        for k in range(len(self.b)):
            for i in range(len(self.b[k])):
                self.params['b'] = b_try
                self.b[k][i] = self.b[k][i] + h
                c2, _ = self.compute_cost(X_batch, Y_batch, lamda)
                self.b[k][i] = self.b[k][i] - 2*h
                c3, _ = self.compute_cost(X_batch, Y_batch, lamda)
                grad_num['b'][k][i] = (c2-c3) / (2*h)
            grad_num['b'][k] = np.squeeze(grad_num['b'][k],axis=1)

        for k in range(len(self.W)):
            for i in np.ndindex(self.W[k].shape):
                self.W = W_try
                self.W[k][i] = self.W[k][i] + h
                c2, _ = self.compute_cost(X_batch, Y_batch, lamda)
                self.W[k][i] = self.W[k][i] - 2*h
                c3, _ = self.compute_cost(X_batch, Y_batch, lamda)
                grad_num['W'][k][i] = (c2-c3) / (2*h)

        return grad_num
    
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
        W_norm = []
        n_b = X.shape[1] # number of images in dataset
        
        
        # [P,_] = self.evaluate_classifier([], X, self.k_num-1)
        P = []
        X = [X]
        # for l in range(0,self.k_num):
        [P, X] = self.evaluate_classifier(X)
        l_cross_sum = np.sum(-Y*np.log(P))
        L =  1/n_b * l_cross_sum                        # total loss
        for i in range(len(self.W)):
            W_norm.append(np.power(np.linalg.norm(self.W[i]),2))
        r = lamda*np.sum(W_norm)                        # regularization term
        J = L + r                                       # cost function of cross-entropy loss
        return [L, J]

    def compute_accuracy(self, X, y):
        """ Computes the accuracy of the classifier
        Args:
            X (np.ndarray): data matrix (D, N)
            y (np.ndarray): labels vector (N)
        Returns:
            acc    (float): the accuracy of the classification on the given data matrix X
        """
        
        P = []
        X = [X]
        [P, X] = self.evaluate_classifier(X)       
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