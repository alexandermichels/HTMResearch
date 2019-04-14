#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 08:52:49 2019

@author: dijkstra
"""

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import pandas as pd
import csv

def gen_histogram(model, order, coef, term):
    data = np.genfromtxt('data/{}{}diffs{}{}.csv'.format(model, order, coef, term), delimiter='\n') # "{type}diffslag{coef}.csv"
    mean = round(np.mean(data), 2)
    stddev = round(np.std(data), 2)

    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=data, bins="auto", density=True, color='#0504aa', alpha=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.title('Error Fitting Lag-{} {} Term of {} Order {} Model'.format(coef, term, order, model))
    plt.text(.2, 6.5, r'$\mu$={}, $\sigma$={}'.format(mean, stddev))
    # Set a clean upper y-axis limit.
    plt.show()
    #plt.savefig("img/{}Model{}Lag{}CoefDist.png".format(model, term, coef))

def get_summary_stats(terms, extra_terms):
    means, stds = [], []
    for model in ["AR", "MA"]:
        for coef in range(1,terms+extra_terms+1):
            data = np.genfromtxt('data/{}{}diffs{}{}.csv'.format(model, terms, model, coef), delimiter='\n')
            mean = round(np.mean(data), 5)
            stddev = round(np.std(data), 5)
            print(model, model, coef)
            means.append(mean)
            print("Mean: {}".format(mean))
            stds.append(stddev)
            print("STDDEV: {}\n".format(stddev))

        other = ""
        if model == "AR":
            other = "MA"
        else:
            other = "AR"
        for coef in range(1,extra_terms+1):
            data = np.genfromtxt('data/{}{}diffs{}{}.csv'.format(model, terms, other, coef), delimiter='\n')
            mean = round(np.mean(data), 5)
            stddev = round(np.std(data), 5)
            print(model, other, coef)
            means.append(mean)
            print("Mean: {}".format(mean))
            stds.append(stddev)
            print("STDDEV: {}\n".format(stddev))
    return means, stds

def gen_histogram_coef(file):
    data = np.genfromtxt('data/{}'.format(file), delimiter='\n') # "{type}diffslag{coef}.csv"
    mean = round(np.mean(data), 2)
    stddev = round(np.std(data), 2)

    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=data, bins="auto", density=True, color='#aa0405', alpha=0.7)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('RMSE')
    plt.ylabel('Density')
    plt.title('RMSE for a 4-Lag AR Instance vs. Instances with\nCoefficients Fitted from the Original Sequence')
    plt.text(-.2, 6.5, r'$\mu$={}, $\sigma$={}'.format(mean, stddev))
    # Set a clean upper y-axis limit.
    plt.show()

def gen_scatterplot(file):
    x, y = [], []
    with open("data/{}".format(file)) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            # print(row)
            x.append(row[0])
            y.append(row[1])
    x = [float(i) for i in x]
    y = [float(i) for i in y]
    # print(x)
    # print(y)
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.scatter(x,y, alpha=0.7, edgecolors='r')
    plt.xlabel('Input Sigma')
    plt.ylabel('Measured Sigma')
    # Set a clean upper y-axis limit.
    fig = plt.gcf()
    fig.set_size_inches(6,5)
    fig.tight_layout()
    fig.savefig('img/SigmaOfSeries.png', dpi=100)
    fig.show()

def gen_scatterplot_from_csv(file, x_header, y_header):
    x, y = [], []
    with open("data/{}".format(file)) as csvfile:
        df = pd.read_csv(csvfile)
    for column in df:
        print(column)
        print(np.mean([float(i) for i in df[column]]))
        print(np.std([float(i) for i in df[column]]))
    x = [float(i) for i in df[x_header]]
    y = [float(i) for i in df[y_header]]
    plt.scatter(x,y, alpha=0.7, edgecolors='r')
    plt.xlabel(x_header)
    plt.ylabel(y_header)
    # Set a clean upper y-axis limit.
    fig = plt.gcf()
    fig.set_size_inches(6,5)
    fig.tight_layout()
    fig.savefig('img/{}by{}.png'.format(x_header, y_header), dpi=100)
    plt.show()

def generate_line_graph(series, name):
    x = range(len(series))
    plt.plot(x,series)
    plt.xlabel('Iteration')
    plt.ylabel('Population Best RMSE')
    fig = plt.gcf()
    fig.set_size_inches(8, 5.5)
    fig.tight_layout()
    fig.show()
    fig.savefig('img/{}-pbest.png'.format(name), dpi=100)


def plot_pbest():
    AR3Full = [0.115296488283918, 0.114422543400309, 0.114031023713655,0.111160555120539,0.09362767205736,0.09362767205736,0.09362767205736,0.09362767205736,0.09362767205736,0.09362767205736,0.083781507999685,0.081589748448596,0.081589748448596,0.081589748448596,0.081589748448596,0.081589748448596,0.081589748448596,0.081589748448596,0.081589748448596,0.081589748448596,0.081589748448596,0.081589748448596,0.081589748448596,0.081589748448596]
    generate_line_graph(AR3Full, "AR3Full")


def main():
    #gen_scatterplot("SigmaOfSeries-PreScaling.csv")
    #means, stds = get_summary_stats(6, 2)
    #confidence_intervals(means, stds)
    gen_scatterplot_from_csv("ScalingGeneralityTestNonScaledRMSE.csv", "Scaling Factor", "Post-Scaling Learning RMSE")
    # Pre-Scaling RMSE	Scaling Factor	Post-Scaling No Learning RMSE	Post-Scaling Learning RMSE


if __name__ == "__main__":
    main()
