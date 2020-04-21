#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:07:15 2020

@author: Alexander Polidar
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

""" Collection of some functions 

    Fns
        Loading
        Normalization
        Plotting

"""

def LoadBatch(filename):
    """ Copied from the dataset website """
    from scipy.io import loadmat
    from os.path import expanduser
    from pathlib import Path
    home = expanduser("~")
    home = str(Path.home())
    with open(home + '/anaconda3/envs/standard/DD2424/git/DD2424/datasets/cifar-10-batches-mat/'+filename, 'rb') as fo:
        dict = loadmat(fo)    
        X = np.transpose(dict['data'])
        y = np.transpose(dict['labels'])
        Y = np.zeros([10,X.shape[1]])
        for i in range(X.shape[1]):
            Y[y[0,i],i] = 1
    return [X,Y,y]

def LoadLabelNames():
    from scipy.io import loadmat
    from os.path import expanduser
    from pathlib import Path
    home = expanduser("~")
    home = str(Path.home())
    with open(home + '/anaconda3/envs/standard/DD2424/git/DD2424/datasets/cifar-10-batches-mat/' + 'batches.meta.mat', 'rb') as fo:
        dict = loadmat(fo) 
        label_name_dict = dict['label_names']
        label_names = {}
        for i in range(label_name_dict.shape[0]):
            label_names[i] = label_name_dict[i][0][0]
    return label_names

def LoadBatchNormalizeSave(filename):
    """ Copied from the dataset website """
    import scipy.io as sio
    from os.path import expanduser
    from pathlib import Path
    home = expanduser("~")
    home = str(Path.home())
    with open(home  + '/anaconda3/envs/standard/DD2424/git/DD2424/datasets/cifar-10-batches-mat/'+filename, 'rb') as fo:
        dict = sio.loadmat(fo)    
        X = dict['data']
        X = Normalize(X)
        dict['data'] = X
        sio.savemat(home + '/anaconda3/envs/standard/DD2424/git/DD2424/datasets/cifar-10-batches-mat/' + 'norm_' + filename, dict)

def Normalize(X):
    mean_X = np.mean(X, axis=1)
    std_X = np.std(X, axis=1)
    
    X = X - np.transpose(np.matlib.repmat(mean_X, X.shape[1], 1))
    X = X / np.transpose(np.matlib.repmat(std_X, X.shape[1], 1))
    return X

def define_layers(shapes, activations):
    layers = OrderedDict([])   
    for i, (shapes, activations) in enumerate(zip(shapes, activations)):
        layers["layer%s" % i] = {"shape": shapes, "activation": activations}
    return layers    

def montage(typ,W,range1,range2, num_set, name, Class, savefig = False):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    from os.path import expanduser
    from pathlib import Path
    home = expanduser("~")
    home = str(Path.home())
    if typ == "photo":
        fig, ax = plt.subplots(range1,range2)
        for i in range(range1):
            for j in range(range2):   
                im  = W[:,5*i+j].reshape(32,32,3, order='F')
                sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
                sim = sim.transpose(1,0,2)
                ax[i][j].imshow(sim, interpolation='nearest')
                ax[i][j].set_title("y="+str(5*i+j))
                ax[i][j].axis('off')
    if typ == "weight":        
        fig, ax = plt.subplots(range2,range1)
        fig.canvas.set_window_title(str(num_set) + "_" + typ + "_" + name)
        for i in range(range1):
            im  = W[i,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i].imshow(sim, interpolation='nearest')
            ax[i].set_title(Class[i], fontsize=6)
            ax[i].axis('off')
    plt.show()
    if savefig:
        plt.savefig(home  + '/anaconda3/envs/standard/DD2424/git/DD2424/assignment2/plots/' + str(num_set) + "_" + typ + "_" + name + ".pdf",
                    bbox_inches="tight")

def plot(figname, num_set, xdata, ydata, label_is, xlim_is, ylim_is, xlabel_is, ylabel_is, savefig = False):
    from os.path import expanduser
    from pathlib import Path
    home = expanduser("~")
    home = str(Path.home())
    plt.figure(str(num_set) + "_" + figname)
    for i in range(len(ydata)):
        plt.plot(xdata, ydata[i], label = label_is[i])
#    plt.xlim(xlim_is)
#    plt.ylim(ylim_is)    
    plt.xlabel(xlabel_is)
    plt.ylabel(ylabel_is)
    plt.legend()
    plt.grid(b=True)
    if savefig:
        plt.savefig(home  + '/anaconda3/envs/standard/DD2424/git/DD2424/assignment2/plots/' + str(num_set) + "_" + figname + ".pdf",
                    bbox_inches="tight")

def save_as_mat(data, name, dic):
	""" Used to transfer a python model to matlab """
	import scipy.io as sio
	sio.savemat('test_save', {'X':X, 'Y':Y, 'y':y})