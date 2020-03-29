#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 00:11:19 2020

@author: alex
"""
import matplotlib.pyplot as plt


def plot(figname, num_set, xdata, ydata, label_is, xlim_is, ylim_is, xlabel_is, ylabel_is):
    plt.figure(figname + str(num_set))
    for i in range(len(ydata)):
        plt.plot(xdata, ydata[i], label = label_is[i])
#    plt.xlim(xlim_is)
#    plt.ylim(ylim_is)    
    plt.xlabel(xlabel_is)
    plt.ylabel(ylabel_is)
    plt.legend()
    plt.grid(b=True)
    