# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import collections
import matplotlib.pyplot as plt

def LoadText(filename):
    f = open(filename, "r")
    text_data = f.read()
    print(f.read())
    return text_data

def CreateDataDict(text_data):
    text_chars = list(set(text_data))
    text_chars.sort()
    char2ind = {}
    ind2char = {}
    for k in range(len(text_chars)):
        char2ind.update({text_chars[k]: k})
        ind2char.update({k: text_chars[k]})
    data = {"text_data": text_data, "text_chars": text_chars, "char2ind": char2ind, "ind2char": ind2char, "num_chars": len(text_chars)}
    return data

def plot(figname, xdata, ydata, xlabel_is, ylabel_is, savefig = False):
    from os.path import expanduser
    from pathlib import Path
    home = expanduser("~")
    home = str(Path.home())
    plt.figure(figname)
    plt.plot(xdata, ydata)
#    plt.xlim(xlim_is)
#    plt.ylim(ylim_is)    
    plt.xlabel(xlabel_is)
    plt.ylabel(ylabel_is)
    plt.grid(b=True)
    if savefig:
        plt.savefig(home  + '/anaconda3/envs/standard/DD2424/git/DD2424/assignment4/plots/' + figname + ".pdf",
                    bbox_inches="tight")

        
    
