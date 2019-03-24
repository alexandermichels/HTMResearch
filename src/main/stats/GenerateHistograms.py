#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 08:52:49 2019

@author: dijkstra
"""

import matplotlib.pyplot as plt
import numpy as np

def gen_histogram(model, coef, term):
    data = np.genfromtxt('data/{}diffs{}lag{}.csv'.format(model, term, coef), delimiter='\n') # "{type}diffslag{coef}.csv"
    mean = round(np.mean(data), 2)
    stddev = round(np.std(data), 2)
    
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=data, bins="auto", density=True, color='#aa0405', alpha=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Fitting Lag-{} {} Term of 6 Order {} Model'.format(coef, term, model))
    plt.text(.2, 3.6, r'$\mu$={}, $\sigma$={}'.format(mean, stddev))
    # Set a clean upper y-axis limit.
    plt.show()
    #plt.savefig("img/{}Model{}Lag{}CoefDist.png".format(model, term, coef))
    
def get_summary_stats(terms, extra_terms):
    for model in ["AR", "MA"]:
        for coef in range(1,terms+extra_terms+1):
            data = np.genfromtxt('data/{}diffs{}lag{}.csv'.format(model, model, coef), delimiter='\n')
            mean = round(np.mean(data), 5)
            stddev = round(np.std(data), 5)
            print(model, model, coef)
            print(mean)
            print(stddev)
            
        term = ""
        if model == "AR":
            term = "MA"
        else:
            term = "AR"
        for coef in range(1,extra_terms+1):
            data = np.genfromtxt('data/{}diffs{}lag{}.csv'.format(model, term, coef), delimiter='\n')
            mean = round(np.mean(data), 5)
            stddev = round(np.std(data), 5)
            print(model, term, coef)
            print(mean)
            print(stddev)
    
def main():
    # gen_histogram("AR", 1, "AR")
    get_summary_stats(6, 2)
    
if __name__ == "__main__":
    main()