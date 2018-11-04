#!/usr/bin/env python
import csv

from HTMNetwork import *
from TimeSeriesStream import TimeSeriesStream
from models.ARMAModels import *
from time import localtime, sleep
import numpy as np
from os.path import join
import logging as log

def runHTM(i, time_series, method):
    log.info("Running HTM.....")
    result = HTM(time_series, cellsPerMiniColumn=i)
    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
    _OUTPUT_PATH = "../outputs/HTMErrors-{}-{}-{}.csv".format(DATE, i, time_series_model)
    # this is what the rows of results look like
    #[_model.getBookmark(), series, oneStep, oneStepConfidence*100, fiveStep, fiveStepConfidence*100]
    one_cum_error = 0
    one_last_half_error = 0
    five_cum_error = 0
    five_last_half_error = 0

    for j in range(5):
        result[j].insert(3,"N/A")
        result[j].insert(6,"N/A")

    for j in range(5,len(result)):
        if method=="MSE":
            one_error = (result[j][2]-result[i-1][1])**2
            five_error = (result[j][4]-result[i-5][1])**2
            result[j].insert(3,one_error)
            result[j].insert(6,five_error)
            one_cum_error = one_cum_error + one_error
            five_cum_error = five_cum_error + five_error
            if j >= len(result)/2.0:
                one_last_half_error+=one_error
                five_last_half_error+=five_error

    with open(_OUTPUT_PATH, "w") as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(["Time Step", "Series", "One Step Prediction", "One Step Prediction Error", "One Step Prediction Confidence", "Five Step Prediction", "Five Step Prediction Error", "Five Step Prediction Confidence"])
        for j in range(len(result)):
            print "{0:6}: 1-step: {2:16} (Error {3:4.4}, Conf: {4:4.4}%)\t 5-step: {5:16} (Error {6:4.4}, Conf: {7:4.4}%)".format(*result[j])
            writer.writerow(result[j])

    return (one_cum_error/(len(result)-5), five_cum_error/(len(result)-5), one_last_half_error/(len(result)/2.0), five_last_half_error/(len(result)/2.0))

def generateErrorSurface(time_series, range_of_cpmc, iterations=200, method="MSE"):
    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))

    for i in range_of_cpmc:
        _OUTPUT_FILE = open("../outputs/{}-{}-model-{}-iterations-with-{}-cellsPerMiniColumn".format(DATE,str(time_series),iterations,i), "w+")
        one_errors = np.zeros(iterations, dtype=np.float64)
        five_errors = np.zeros(iterations, dtype=np.float64)
        second_one_errors = np.zeros(iterations, dtype=np.float64)
        second_five_errors = np.zeros(iterations, dtype=np.float64)
        log_file = join('../logs/', 'log_{}_model:{}_{}CPMC.log'.format(DATE, str(time_series), i))
        log.basicConfig(format = '[%(asctime)s] %(message)s', datefmt = '%m/%d/%Y %H:%M:%S %p', filename = log_file, level=log.DEBUG)
        log.getLogger().addHandler(log.StreamHandler())

        for num_iter in range(iterations):
            result = runHTM(i, time_series, method)
            one_errors[num_iter] = result[0]
            five_errors[num_iter] = result[1]
            second_one_errors[num_iter] = result[2]
            second_five_errors[num_iter] = result[3]

            log.info("The average one-step error for iteration {} was {} and the average five-step error was {}".format(num_iter, one_errors[num_iter], five_errors[num_iter]))
            log.info("The second-half average one-step error for iteration {} was {} and the second-half average five-step error was {}".format(num_iter, second_one_errors[num_iter], second_five_errors[num_iter]))
            _OUTPUT_FILE.write("{} {} {}\n".format(num_iter, second_one_errors[num_iter], second_five_errors[num_iter]))
            time_series.new()
        _OUTPUT_FILE.close()

        log.info("The average one-step error over {} iterations was {} with standard deviation {} and the average five-step error was {} with standard deviation {}".format(iterations, np.average(one_errors), np.std(one_errors), np.average(five_errors), np.std(five_errors)))
        log.info("The average second-half one-step error over {} iterations was {} with standard deviation {} and the average second-half five-step error was {} with standard deviation {}".format(iterations, np.average(second_one_errors), np.std(second_one_errors), np.average(second_five_errors), np.std(second_five_errors)))

if __name__ == "__main__":
    time_series_model = ARMATimeSeries(2,0)
    generateErrorSurface(time_series_model, range(2,12))
