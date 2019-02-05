#!/usr/bin/env python
import csv

from HTMNetwork import *
from TimeSeriesStream import TimeSeriesStream
from models.ARMAModels import *
from models.SimpleSequence import *

from time import localtime, sleep
import itertools
import numpy as np
from os.path import join
from tqdm import tqdm
import logging as log
import multiprocessing as mp

DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))

def runHTM(i, time_series, method):
    log.info("Running HTM.....")
    network = HTM(time_series, .1, cellsPerMiniColumn=i, verbosity=0)
    result = train(network, "expressive", method)

    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
    _OUTPUT_PATH = "../outputs/HTMErrors-{}-{}-{}.csv".format(DATE, i, time_series_model)
    # this is what the rows of results look like
    #[_model.getBookmark(), series, oneStep, oneStepConfidence*100, fiveStep, fiveStepConfidence*100]
    one_cum_error = 0
    five_cum_error = 0

    for j in range(5):
        result[j].insert(3,"N/A")
        result[j].insert(6,"N/A")

    for j in range(5,len(result)):
        if method=="MSE":
            one_error = (result[j-1][2]-result[j-1][1])**2
            five_error = (result[j-5][4]-result[j][1])**2
        elif method=="Binary":
            if result[j-1][2] == result[j][1]:
                one_error = 0
            else:
                one_error = 1
            if result[j-5][5] == result[j][1]:
                five_error = 0
            else:
                five_error = 1
        #record errors
        result[j].insert(3,one_error)
        result[j].insert(6,five_error)
        one_cum_error += one_error
        five_cum_error += five_error

    with open(_OUTPUT_PATH, "w") as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(["Time Step", "Series", "One Step Prediction", "One Step Prediction Error", "One Step Prediction Confidence", "Five Step Prediction", "Five Step Prediction Error", "Five Step Prediction Confidence"])
        for j in range(len(result)):
            #print "{0:6}: 1-step: {2:16} (Error {3:4.4}, Conf: {4:4.4}%)\t 5-step: {5:16} (Error {6:4.4}, Conf: {7:4.4}%)".format(*result[j])
            writer.writerow(result[j])

    return (one_cum_error/(len(result)-5), five_cum_error/(len(result)-5), i)

def generateErrorSurface(time_series, range_of_cpmc, iterations=200, method="MSE"):
    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
    log_file = join('../logs/', 'log_{}-{}-model-{}-iterations-with-({}-{})-cellsPerMiniColumn.log'.format(DATE,str(time_series),iterations,min(range_of_cpmc),max(range_of_cpmc)))

    with open(join("../outputs/", 'csv_{}-{}-model-{}-iterations-with-({}-{})-cellsPerMiniColumn.csv'.format(DATE,str(time_series),iterations,min(range_of_cpmc),max(range_of_cpmc))), "w+") as csv_out:
        writer = csv.writer(csv_out)

        #write header row:
        header_row = []
        for i in range_of_cpmc:
            header_row.append("{} CPMC One-Step Error".format(i))
            header_row.append("{} CPMC Five-Step Error".format(i))
        writer.writerow(header_row)

        for num_iter in tqdm(range(iterations)):
            log.basicConfig(format = '[%(asctime)s] %(message)s', datefmt = '%m/%d/%Y %H:%M:%S %p', filename = log_file, level=log.DEBUG)
            log.getLogger().addHandler(log.StreamHandler())

            output_row = []
            for i in range_of_cpmc:
                result = runHTM(i, time_series, method)
                time_series.rewind()
                output_row.append(result[2])
                output_row.append(result[3])

            writer.writerow(output_row)
            time_series.new()

def runHTMPar(i, time_series, method):
    log.info("Running HTM.....")
    network = HTM(time_series, .1, cellsPerMiniColumn=i, verbosity=0)
    result = train(network, "expressive", method)

    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
    _OUTPUT_PATH = "../outputs/HTMErrors-{}-{}-{}.csv".format(DATE, i, time_series_model)
    # this is what the rows of results look like
    #[_model.getBookmark(), series, oneStep, oneStepConfidence*100, fiveStep, fiveStepConfidence*100]
    one_cum_error = 0
    five_cum_error = 0

    for j in range(5):
        result[j].insert(3,"N/A")
        result[j].insert(6,"N/A")

    for j in range(5,len(result)):
        if method=="MSE":
            one_error = (result[j-1][2]-result[j-1][1])**2
            five_error = (result[j-5][4]-result[j][1])**2
        elif method=="Binary":
            if result[j-1][2] == result[j][1]:
                one_error = 0
            else:
                one_error = 1
            if result[j-5][5] == result[j][1]:
                five_error = 0
            else:
                five_error = 1
        #record errors
        result[j].insert(3,one_error)
        result[j].insert(6,five_error)
        one_cum_error += one_error
        five_cum_error += five_error


    with open(_OUTPUT_PATH, "w") as outputFile:
        _writer = csv.writer(outputFile)
        _writer.writerow(["Time Step", "Series", "One Step Prediction", "One Step Prediction Error", "One Step Prediction Confidence", "Five Step Prediction", "Five Step Prediction Error", "Five Step Prediction Confidence"])
        for j in range(len(result)):
            #print "{0:6}: 1-step: {2:16} (Error {3:4.4}, Conf: {4:4.4}%)\t 5-step: {5:16} (Error {6:4.4}, Conf: {7:4.4}%)".format(*result[j])
            _writer.writerow(result[j])

    return (i, (one_cum_error/float(len(result)-5), five_cum_error/float(len(result)-5)))

def runHTMParUnpacker(args):
    return runHTMPar(*args)

def generateErrorSurfacePar(time_series, range_of_cpmc, iterations=200, method="MSE"):
    log_file = join('../logs/', 'log_{}-{}-model-{}-iterations-with-({}-{})-cellsPerMiniColumn.log'.format(DATE,str(time_series),iterations,min(range_of_cpmc),max(range_of_cpmc)))

    with open(join("../outputs/", 'csv_{}-{}-model-{}-iterations-with-({}-{})-cellsPerMiniColumn.csv'.format(DATE,str(time_series),iterations,min(range_of_cpmc),max(range_of_cpmc))), "w+") as csv_out:
        writer = csv.writer(csv_out)

        #write header row:
        header_row = []
        for i in range_of_cpmc:
            header_row.append("{} CPMC One-Step Error".format(i))
            header_row.append("{} CPMC Five-Step Error".format(i))
        writer.writerow(header_row)

        for num_iter in tqdm(range(iterations)):
            log.basicConfig(format = '[%(asctime)s] %(message)s', datefmt = '%m/%d/%Y %H:%M:%S %p', filename = log_file, level=log.DEBUG)
            log.getLogger().addHandler(log.StreamHandler())

            p = mp.Pool(processes = mp.cpu_count()-1)
            results = p.map(runHTMParUnpacker, itertools.izip(range_of_cpmc, itertools.repeat(time_series), itertools.repeat(method)))

            #input_queue = mp.Queue()
            #processes = [mp.Process(target=runHTMPar, args=(i, time_series, method, input_queue)) for i in range_of_cpmc]

            #for p in processes:
            #    p.start()
            #for p in processes:
            #    p.join()

            #results = [input_queue.get() for p in processes]
            results.sort()
            for r in results:
                log.info("For {} CPMC the one-step error was {} and the five-step error was {}".format(r[0], r[1][0], r[1][1]))
            results = [r[1] for r in results]

            output_row = []
            for r in results:
                output_row.append(r[0])
                output_row.append(r[1])
            writer.writerow(output_row)
            time_series.new()

if __name__ == "__main__":
    time_series_model = VeryBasicSequence()
    generateErrorSurfacePar(time_series_model, range(3,8), method="Binary")
