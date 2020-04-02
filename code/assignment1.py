#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:55:51 2020

@author: Alexander Polidar
"""

import numpy as np
import numpy.matlib
import functions_base as fb
import ClassifierMiniBatchGD as CMBGD
import unittest

###############################################################################
"""SETUP"""
###############################################################################

""" Settings 
        n_it:           no. of iterations of mini-batch gradient descent: determining mean and std of accuracies
        
        set2eval:       choose datasets to load and evaluate, if len(set2eval) > 1: stacking of datasets
        GDparams2eval:  choose parameter sets to evaluate
        
        n_batch:        number of batches in dataset
        n_epoch:        number of epochs for which mini-batch gradient descent is ran
        
        plot_results    set whether results (total loss, cost, accuracies, weight matrices) should be plotted (0/1)
        plot_photos     set whether some data (photos) should be plotted (0/1)
        export_results  set whether results should be exported
                
        GDparams:       different sets of parameters of gradient descent: [lambda: regulization parameter, eta: step-size for gradient descent]       
"""

# no. of iterations
n_it = 10

# choose sets and parameters to evaluate
set2eval = [1]  
GDparams2eval = [0,1,2,3]#[0, 1, 2, 3]     # choose with which paramter sets to run

# mini-batch gradient descent
n_batch = 100
n_epoch = 40

# plotting of results
plot_results = 1
plot_photos = 0
export_results = 0

# parameter settings
GDparams = [[0,     .1],
            [0,     .001],
            [.1,    .001],
            [1,     .001]]


###############################################################################
"""Pre-definitions"""
###############################################################################

Name = ['Training set', 'Validation set', 'Dataset 3', 'Dataset 4', 'Dataset 5'] 

L_train = np.zeros((n_epoch, len(GDparams2eval)))
J_train = np.zeros((n_epoch, len(GDparams2eval)))
L_val = np.zeros((n_epoch, len(GDparams2eval)))
J_val = np.zeros((n_epoch, len(GDparams2eval)))

acc_train = np.zeros((n_epoch, len(GDparams2eval)))
acc_val = np.zeros((n_epoch, len(GDparams2eval)))
acc_test = np.zeros((n_epoch, len(GDparams2eval)))
acc_train_end = np.zeros((n_it, len(GDparams2eval)))
acc_val_end = np.zeros((n_it, len(GDparams2eval)))
acc_test_end = np.zeros((n_it, len(GDparams2eval)))

acc_train_mean = list([])
acc_val_mean = list([])
acc_test_mean = list([])

acc_train_std = list([])
acc_val_std = list([])
acc_test_std = list([])


train_acc = np.zeros((len(GDparams),n_it))
train_acc[:] = np.nan
test_acc = np.zeros((len(GDparams),n_it))
test_acc[:] = np.nan
val_acc = np.zeros((len(GDparams),n_it))
val_acc[:] = np.nan
            
###############################################################################
""" Load data and stack if wanted """
###############################################################################

for s in range(len(set2eval)):
    print('\r' + str(Name[s]) + '\n', flush=True, end='')
           
    if len(set2eval) == 1:
        [X_train,Y_train,y_train] = fb.LoadBatch('norm_data_batch_' + str(set2eval[s]) + '.mat')
    elif len(set2eval) > 1:
        if s == 0:
            [X_train,Y_train,y_train] = fb.LoadBatch('norm_data_batch_' + str(set2eval[s]) + '.mat')
        else:
            [X_train_set, Y_train_set, y_train_set] = fb.LoadBatch('norm_data_batch_' + str(set2eval[s]) + '.mat')
            X_train = np.append(X_train, X_train_set, axis = 1)
            Y_train = np.append(Y_train, Y_train_set, axis = 1)
            y_train = np.append(y_train, y_train_set, axis = 1)     
    
if len(set2eval) == 1:
    [X_val,Y_val,y_val] = fb.LoadBatch('norm_data_batch_2.mat')        
else:   
    N_val = 1000
    [X_val,Y_val,y_val] = [X_train[:,-N_val:],Y_train[:,-N_val:],y_train[:,-N_val:]]
    [X_train,Y_train,y_train] = [X_train[:,:-N_val],Y_train[:,:-N_val],y_train[:,:-N_val]]
    
[X_test,Y_test,y_test] = fb.LoadBatch('norm_test_batch.mat')   

###############################################################################
""" Setup classifier """
###############################################################################

data = {
    'X_train': X_train,
    'Y_train': Y_train,
    'y_train': y_train,
    'X_val': X_val,
    'Y_val': Y_val,
    'y_val': y_val,
    'X_test': X_test,
    'Y_test': Y_test,
    'y_test': y_test
}

label_names = fb.LoadLabelNames()

clfr = CMBGD.ClassifierMiniBatchGD(data, label_names)

###############################################################################
""" Unittest classifier """
###############################################################################
    
if __name__ == '__main__':            
    class Testing(unittest.TestCase):   
        clfr = CMBGD.ClassifierMiniBatchGD(data, label_names)
        
        def test_0_sizes(self):
            np.testing.assert_equal(np.shape(self.clfr.X_train), (3072, 10000), err_msg='Training data')
            np.testing.assert_equal(np.shape(self.clfr.Y_train), (10, 10000), err_msg='One-hot-encoded label-matrix')
            np.testing.assert_equal(np.shape(self.clfr.y_train), (1,10000), err_msg='Labels')
            np.testing.assert_equal(np.shape(self.clfr.softmax(X_train)), (10,10000), err_msg='Output of softmax: Probability matrix P')
            np.testing.assert_almost_equal(np.sum(self.clfr.softmax(X_train), axis=0), 1, decimal = 6, err_msg='Sum of probabilities != 1')                
            
        def test_1_gradients(self):
            self.clfr.W = self.clfr.W[:,:20]
            [grad_W, grad_b] = self.clfr.compute_gradients(self.clfr.X_train[:20,:100], self.clfr.Y_train[:,:100], lamda = 0)
            [grad_W_num, grad_b_num] = self.clfr.compute_gradients_num(self.clfr.X_train[:20,:100], self.clfr.Y_train[:,:100], lamda = 0)
            np.testing.assert_almost_equal(grad_W, grad_W_num, decimal = 6, err_msg='Gradient calulcation (num/ana): W matrix not almost equal up to 6 decimals')
            np.testing.assert_almost_equal(grad_b, grad_b_num, decimal = 6, err_msg='Gradient calulcation (num/ana): b vector not almost equal up to 6 decimals')
            error_rel_W_max = np.max(np.abs(grad_W - grad_W_num)/np.maximum(1e-6, np.abs(grad_W) + np.abs(grad_W_num)))
            error_rel_b_max = np.max(np.abs(grad_b - grad_b_num)/np.maximum(1e-6, np.abs(grad_b) + np.abs(grad_b_num)))
            print('Maximum relative error between numerical and analytical calculation of W: ' + str("{:.6f}".format(error_rel_W_max*100)) + ' %')
            print('Maximum relative error between numerical and analytical calculation of b: ' + str("{:.6f}".format(error_rel_b_max*100)) + ' %')
                
    # Unit testing
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
    
###############################################################################
""" Conduct mini-batch gradient descent """
###############################################################################

for i in range(len(GDparams2eval)):
    print('\r    Parameter set ' + str(GDparams2eval[i]) + '\n', flush=True, end='')
    
    # initialize each parameter set with the same random distribution
    np.random.seed(0)
    
    for n in range(n_it):
        clfr = CMBGD.ClassifierMiniBatchGD(data, label_names)
        [W_star, acc_train[:,i], acc_val[:,i], acc_test[:,i], L_train[:,i], J_train[:,i], L_val[:,i], J_val[:,i]] = clfr.mini_batch_gd(
                                                                    X_train,
                                                                    Y_train,
                                                                    y_train,
                                                                    GDparams[GDparams2eval[i]][0],
                                                                    GDparams[GDparams2eval[i]][1],
                                                                    n_batch,
                                                                    n_epoch,
                                                                    eval_performance = True)
        
        [acc_train_end[n,i], acc_val_end[n,i], acc_test_end[n,i]]= [acc_train[-1,i], acc_val[-1,i], acc_test[-1,i]]
    
    # Calculate mean of accuracies w.r.t. no. of iterations
    acc_train_mean.append(np.mean(acc_train_end[:,i], axis=0))
    acc_val_mean.append(np.mean(acc_val_end[:,i], axis=0))
    acc_test_mean.append(np.mean(acc_test_end[:,i], axis=0))

    # Calculate std of accuracies w.r.t. no. of iterations            
    acc_train_std.append(np.std(acc_train_end[:,i], axis=0))
    acc_val_std.append(np.std(acc_val_end[:,i], axis=0))
    acc_test_std.append(np.std(acc_test_end[:,i], axis=0))
    
    print("\n        Classification accuracy on training set: " + str("{:.2f}".format(100*acc_train_mean[i])) + " +- " + str("{:.2f}".format(100*acc_train_std[i])) + " %")
    print("        Classification accuracy on validation set: " + str("{:.2f}".format(100*acc_val_mean[i])) + " +- " + str("{:.2f}".format(100*acc_val_std[i])) + " %")
    print("        Classification accuracy on test set: " + str("{:.2f}".format(100*acc_test_mean[i])) + " +- " + str("{:.2f}".format(100*acc_test_std[i])) + " %")
    
    ###########################################################################
    """ Plot results """
    ###########################################################################     
    
    if plot_results:            
        fb.plot("total_loss", 
                str(GDparams2eval[i]), 
                np.arange(L_train.shape[0])+1,
                [L_train[:,i], L_val[:,i]], 
                [Name[s], Name[1]], 
                (1,GDparams[GDparams2eval[i]][1]), 
                (1.5,2.5), 
                'Number of epochs',
                'Total loss',
                savefig = export_results)
        
        fb.plot("cost_function",
                str(GDparams2eval[i]),
                np.arange(L_train.shape[0])+1,
                [J_train[:,i], J_val[:,i]],
                [Name[s], Name[1]],
                (1,GDparams[GDparams2eval[i]][1]),
                (1.5,2.5),
                'Number of epochs',
                'Cost function',
                savefig = export_results)
        
        fb.plot("accuracies",
                str(GDparams2eval[i]),
                np.arange(L_train.shape[0])+1,
                [acc_train[:,i], acc_val[:,i], acc_test[:,i]],
                ['Training set', 'Validation set', 'Test set'],
                (1,GDparams[GDparams2eval[i]][1]),
                (1.5,2.5),
                'Number of epochs',
                'Accuracy',
                savefig = export_results)    
        
        fb.montage("weight", W_star, 10, 1, GDparams2eval[i], Name[s], label_names, savefig = export_results)
        


###############################################################################
""" Plot CIFAR-10 photos """
###############################################################################
        
""" Show photos"""
if plot_photos:
    fb.montage('photo', X_train, 3, 5, [], [], [])     