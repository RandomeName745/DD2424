#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:27:06 2020

@author: alex
"""

import functions_base as fb
import numpy as np

import TextSynthesizer as TS


###############################################################################
""" SETUP """
###############################################################################
""" Settings 
    m (int): dimensionality of hidden states
    eta (float): learning rate
    seq_length (int): length of the input sequence
    sig (float): factor for initialization
"""
# Define hyper-parameters
m = 100
eta = .1
seq_length = 25

# Define initialization of weight matrices
sig = .01

###############################################################################
""" Load data text and create data dictionary """
###############################################################################
text_data = fb.LoadText("/home/alex/anaconda3/envs/standard/DD2424/git/DD2424/assignment4/goblet_book.txt")
data = fb.CreateDataDict(text_data)

###############################################################################
""" Synthesize text """
###############################################################################

txtsynth = TS.TextSynthesizer(data, m, eta, seq_length, sig)
smooth_loss, step = txtsynth.RNN_SGD()
fb.plot("smooth_loss", np.array(range(0,step)), smooth_loss[1:], "Number of update steps", "Smooth loss")





