#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:55:51 2020

@author: alex
data"""
import numpy as np
import numpy.matlib
#import matplotlib.pyplot as plt
import plot as plot
import functions_base as fb
import miniBatchGD as mBGD

""" Settings 
        n_set:          no. of datasets to load and combine
        eval_set:       choose datasets to load
        val_grad:       validate gradients? (0/1)
        GDparams:       different sets of parameters of gradient descent: [batch length, no. of epochs, step-size for gradient descent, lambda for regulization term]
        n_GDparams:     no. of sets of parameters to do the gradient descent method for
        eval_GDparams:  choose parameter sets to evaluate
        plot_data       decide whether some data (photos) should be pltted (0/1)
        Name:           names of datasets (for plotting)
"""
n_it = 1
#n_set = 1
eval_set = [1]  
val_grad = 0   
#n_GDparams = 4
eval_GDparams = [1]#[0, 1, 2, 3]     # choose with which paramter sets to run
plot_data = 0

GDparams = [[100, 40, .1, 0],
                [100, 40, .001, 0],
                [100, 40, .001, .1],
                [100, 40, .001, 1]]

Name = ['Training set', 'Validation set', 'Dataset 3', 'Datsaet 4', 'Dataset 5'] 
Class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_acc = np.zeros((len(GDparams),n_it))
train_acc[:] = np.nan
test_acc = np.zeros((len(GDparams),n_it))
test_acc[:] = np.nan
val_acc = np.zeros((len(GDparams),n_it))
val_acc[:] = np.nan
    
for n in range(n_it):
    
    """ Load data """
    
    for s in range(len(eval_set)):
        print('\r' + str(Name[s]) + '\n', flush=True, end='')
        [X,Y,y] = fb.LoadBatch('norm_data_batch_' + str(eval_set[s]) + '.mat')
        [X_val,Y_val,y_val] = fb.LoadBatch('data_batch_2.mat')
        [X_test,Y_test,y_test] = fb.LoadBatch('test_batch.mat')
        if plot_data == 1:
            fb.montage('photo', X, 3, 5, [], [], [])
        
        """ for stacking data sets """
        #    if s == 0:
        #        X = X_set
        #        Y = Y_set
        #        y = y_set
        #    else:
        #        X = np.append(X, X_set, axis = 1)
        #        Y = np.append(Y, Y_set, axis = 1)
        #        y = np.append(y, y_set, axis = 1)
                
        """ Pre-process data: normalization w.r.t. mean, std """
        
        X = fb.Normalize(X)
        X_val = fb.Normalize(X_val)
        X_test = fb.Normalize(X_test)
        
        """ Initialize W and b with mean = 0, std = 0.01, size=(dim0: # of labels, dim1: pixel data)"""
            
        np.random.seed(0)
        W_init = np.random.normal(loc=0.0, scale=0.01, size=(Y.shape[0],X.shape[0]))
        np.random.seed(0)
        b_init = np.random.normal(loc=0.0, scale=0.01, size=(Y.shape[0],1))
        b_init = np.squeeze(b_init,axis=1)
        
        """ Pre-definitions """
        
        if val_grad == 1:
            n_dim = np.arange(20) # see: data[:split,:]
            
        if val_grad == 1:
            W = W_init[:,n_dim]                                     # initialize parameters for optimization
        else:
            W = W_init
            
        b = b_init  
           
        
        """ Batch grdient descent method
                z-loop: parameter-loop
                j-loop: epoch-loop, GDparams[z][1]: no. of epochs
                i-loop: batch-loop, (X_perm.shape[1] / GDparams[z][0]): no. of batches
        """
                        
        for z in range(len(eval_GDparams)): 
            print('\r    Parameter set ' + str(eval_GDparams[z]) + '\n', flush=True, end='')
            J = np.array([])
            J_val = np.array([])
            L = []
            L_val = []
            for j in range(GDparams[eval_GDparams[z]][1]):
                randperm = np.random.permutation(X.shape[1])                            # shuffle training data
                X_perm = X[:,randperm]
                Y_perm = Y[:,randperm]
                y_perm = y[:,randperm]
#                X_perm = X
#                Y_perm = Y
#                y_perm = y
                for i in range(1,int(X_perm.shape[1] / GDparams[eval_GDparams[z]][0])):
                    print('\r        Epoch ' + str(j+1) + ' - progress: ' + str(100*(i+1)/(X_perm.shape[1] / GDparams[eval_GDparams[z]][0])) + " %", flush=True, end='')
                    i_start = (i-1) * GDparams[eval_GDparams[z]][0]
                    i_end = i * GDparams[eval_GDparams[z]][0]
                    idx = np.arange(i_start, i_end)
                    if val_grad == 1:
                        X_batch = X_perm[n_dim, :]
                        X_batch = X_batch[:, idx]
                    else:
                        X_batch = X_perm[:, idx]
                    Y_batch = Y_perm[:, idx]   
                    y_batch = y_perm[:, idx]     
                    train_batch = mBGD.MiniBatchGD(X_batch, Y_batch, y_batch, GDparams[eval_GDparams[z]], W, b)  # define batch of training data for gradient-descent
                    [W, b] = train_batch.ComputeParameters() 
                                                     # re-compute optimum parameters for batch: backwards...
                    """ Validate gradients """
                    
                    if val_grad == 1:
                        grad_W = train_batch.grad_W
                        grad_b = train_batch.grad_b
                        [ngrad_W, ngrad_b] = fb.ComputeGradsNumSlow(X_batch, Y_batch, W, b, GDparams[eval_GDparams[z]][3], 1e-6)
                        eps = 1e-15
                        error_rel_W = (np.abs(grad_W - ngrad_W))/np.maximum(eps, np.abs(grad_W) + np.abs(ngrad_W))
                        error_rel_W_max = np.max(error_rel_W)
                        error_rel_W_min = np.min(error_rel_W)
                        error_rel_W_mean = np.mean(error_rel_W)
                        
                """ Calculate losses for each epoch for training data and validation data """
                
                epoch = mBGD.MiniBatchGD(X_perm, Y_perm, y_perm, GDparams[eval_GDparams[z]], W, b)
                epoch.ScoreClassifier()
                epoch.Softmax()
                [epoch_L, epoch_J] = epoch.ComputeCost()
                J = np.concatenate((J, np.array([epoch_J])), axis = 0)
                L = np.concatenate((L, np.array([epoch_L])), axis = 0)   
                epoch_val = mBGD.MiniBatchGD(X_val, Y_val, y_val, GDparams[eval_GDparams[z]], W, b)       
                epoch_val.ScoreClassifier()
                epoch_val.Softmax()
                [epoch_val_L, epoch_val_J] = epoch_val.ComputeCost()
                J_val = np.concatenate((J_val, np.array([epoch_val_J])), axis = 0)
                L_val = np.concatenate((L_val, np.array([epoch_val_L])), axis = 0)
                
            """ Plot losses   
                plot.plot(figname, num_set, xdata, ydata, label_is, xlim_is, ylim_is, xlabel_is, ylabel_is)
            """
            
            plot.plot("Total loss: parameter-set ", str(eval_GDparams[z]), np.arange(L.size)+1, [L, L_val], [Name[s], Name[1]], (1,GDparams[eval_GDparams[z]][1]), (1.5,2.5), 'Number of epochs', 'Total loss')
            plot.plot("Cost function: parameter-set ", str(eval_GDparams[z]), np.arange(L.size)+1, [J, J_val], [Name[s], Name[1]], (1,GDparams[eval_GDparams[z]][1]), (1.5,2.5), 'Number of epochs', 'Cost function')
            if Name[s] == Name[0]:
                fb.montage("weight", W, 10, 1, eval_GDparams[z], Name[s], Class)
            
            """ Compute accuracy w.r.t.
                    whole training dataset
                    test data set
            """
            
            if val_grad == 1:
                train = mBGD.MiniBatchGD(X[n_dim,:], Y, y, GDparams[eval_GDparams[z]], W, b)
            else: 
                train = mBGD.MiniBatchGD(X, Y, y, GDparams[eval_GDparams[z]], W, b)
                test = mBGD.MiniBatchGD(X_test, Y_test, y_test, GDparams[eval_GDparams[z]], W, b)  
                val = mBGD.MiniBatchGD(X_val, Y_val, y_val, GDparams[eval_GDparams[z]], W, b)  
            train_acc[eval_GDparams[z], n] = train.ComputeAccuracy()
            test_acc[eval_GDparams[z], n]  = test.ComputeAccuracy()
            val_acc[eval_GDparams[z], n]  = val.ComputeAccuracy()
            
            print("\n        Classification accuracy on training set: " + str(100*train_acc[eval_GDparams[z]]) + " %")
            print("        Classification accuracy on validation set: " + str(100*val_acc[eval_GDparams[z]]) + " %")
            print("        Classification accuracy on test set: " + str(100*test_acc[eval_GDparams[z]]) + " %")
            
if  Name[s] == 'Training set':
    acc = {'training acc': train_acc, 'testing acc': test_acc, 'validation acc': val_acc}