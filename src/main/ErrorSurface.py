#!/usr/bin/env python
import csv

from HTMNetwork import *
from TimeSeriesStream import TimeSeriesStream
from models.ARMAModels import *
from time import localtime, sleep
import numpy as np
from os.path import join
import logging as log

DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))

def runHTM(i, time_series, method):
    log.info("Running HTM.....")
    result = HTM(time_series, cellsPerMiniColumn=i)
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
            #print "{0:6}: 1-step: {2:16} (Error {3:4.4}, Conf: {4:4.4}%)\t 5-step: {5:16} (Error {6:4.4}, Conf: {7:4.4}%)".format(*result[j])
            writer.writerow(result[j])

    return (one_cum_error/(len(result)-5), five_cum_error/(len(result)-5), one_last_half_error/(len(result)/2.0-5), five_last_half_error/(len(result)/2.0-5))

def generateErrorSurface(time_series, range_of_cpmc, iterations=1, method="MSE"):
    log_file = join('../logs/', 'log_{}-{}-model-{}-iterations-with-({}-{})-cellsPerMiniColumn.log'.format(DATE,str(time_series),iterations,min(range_of_cpmc),max(range_of_cpmc)))

    with open(join("../outputs/", 'csv_{}-{}-model-{}-iterations-with-({}-{})-cellsPerMiniColumn.csv'.format(DATE,str(time_series),iterations,min(range_of_cpmc),max(range_of_cpmc))), "w+") as csv_out:
        writer = csv.writer(csv_out)

        #write header row:
        header_row = []
        for i in range_of_cpmc:
            header_row.append("{} CPMC One-Step Error".format(i))
            header_row.append("{} CPMC Five-Step Error".format(i))
        writer.writerow(header_row)

        for num_iter in range(iterations):
            log.basicConfig(format = '[%(asctime)s] %(message)s', datefmt = '%m/%d/%Y %H:%M:%S %p', filename = log_file, level=log.DEBUG)
            log.getLogger().addHandler(log.StreamHandler())

            output_row = []
            for i in range_of_cpmc:
                result = runHTM(i, time_series, method)
                time_series.rewind()
                output_row.append(result[2])
                output_row.append(result[3])

                log.info("The average one-step error was {} and the average five-step error was {}".format(result[0], result[1]))
                log.info("The second-half average one-step error was {} and the second-half average five-step error was {}".format(result[2], result[3]))
            writer.writerow(output_row)
            time_series.new()

if __name__ == "__main__":
    time_series_model = ARMATimeSeries(2,0)
    generateErrorSurface(time_series_model, range(2,13))
