#!/usr/bin/env python
import itertools
import multiprocessing as mp

from HTM import *
from models.ARMAModels import *

def fitHTMOutputs(ARrange, MArange, CPMCrange, n = 10):
    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
    if MArange == []:
        MArange = [0]
    if MArange == [0]:
        _OUTPUT_PATH = "../outputs/TimeSeriesFitter-{}-ARRange({}-{})-MARange-(NULL)-CPMCRange-({}-{}).csv".format(DATE, min(ARrange), max(ARrange), min(CPMCrange), max(CPMCrange))
    else:
        _OUTPUT_PATH = "../outputs/TimeSeriesFitter-ARRange({}-{})-MARange-({}-{})-CPMCRange-({}-{}).csv".format(DATE, min(ARrange), max(ARrange), min(MArange), max(MArange), min(CPMCrange), max(CPMCrange))
    with open(_OUTPUT_PATH, "w") as outputFile:
        writer = csv.writer(outputFile)
        header_row = ["Run #", "CPMC", "AR Order", "MA Order"]
        for ar in ARrange:
            if not ar == 0:
                header_row.append("AR {}-lag Coeff".format(ar))
        for ma in MArange:
            if not ma == 0:
                header_row.append("MA {}-lag Coeff".format(ma))
        header_row.append("BIC AR Order")
        header_row.append("BIC MA Order")
        for ar in ARrange:
            if not ar == 0:
                header_row.append("Fitted AR {}-lag Coeff".format(ar))
        for ma in MArange:
            if not ma == 0:
                header_row.append("Fitted MA {}-lag Coeff".format(ma))
        header_row.append("HTM Error on Prediction")
        header_row.append("HTM Pred AR Order")
        header_row.append("HTM Pred MA Order")
        for ar in ARrange:
            if not ar == 0:
                header_row.append("HTM AR {}-lag Coeff".format(ar))
        for ma in MArange:
            if not ma == 0:
                header_row.append("HTM MA {}-lag Coeff".format(ar))
        writer.writerow(header_row)
        counter = 0

        for cpmc in CPMCrange:
            for ar in ARrange:
                for ma in MArange:
                    for justdoit in range(n): # parallelize on n
                        result_row = [counter, cpmc, ar, ma]
                        print("{}, {}, {}".format(cpmc, ar, ma))
                        ts = ARMATimeSeries(ar,ma) # generate a (ar,ma)-ARMA model
                        for i in range(1, len(ts.ar_poly)):
                            result_row.append(ts.ar_poly[i])
                        for i in range(max(ARrange)-len(ts.ar_poly)+1): # fit the modelrange_ar-len(ts.ar_poly)):
                            result_row.append(0)
                        for i in range(1, len(ts.ma_poly)):
                            result_row.append(ts.ma_poly[i])
                        for i in range(max(MArange)-len(ts.ma_poly)+1):
                            result_row.append(0)
                        tso = get_order(ts.sequence, max_ar=(max(ARrange)+1), max_ma=(max(MArange)+1)) # use BIC to get order
                        print(tso)
                        result_row.append(tso[0])
                        result_row.append(tso[1])
                        tarps, tmaps = fit(ts.sequence, tso) # fit the model
                        print(tarps, tmaps)
                        for i in range(len(tarps)):
                            if i < max(ARrange):
                                result_row.append(-1*tarps[i])
                        for i in range(max(ARrange)-len(tarps)): # fit the modelrange_ar-len(ts.ar_poly)):
                            result_row.append(0)
                        for i in range(len(tmaps)):
                            if i < max(MArange):
                                result_row.append(-1*tmaps[i])
                        for i in range(max(MArange)-len(tmaps)):
                            result_row.append(0)
                        # train the HTM
                        network = HTM(ts, 5, verbosity=0)
                        result_row.append(network.train("rmse", sibt=0, iter_per_cycle=1, normalize_error=True)) # record its error
                        ones, res = network.runNetwork()
                        tso = get_order(ones, max_ar=(max(ARrange)+1), max_ma=(max(MArange)+1)) # use BIC to get order
                        network.__del__
                        del network
                        print(tso)
                        result_row.append(tso[0])
                        result_row.append(tso[1])
                        tarps, tmaps = fit(ones, tso) # fit the model
                        print(tarps, tmaps)
                        for i in range(len(tarps)):
                            if i < max(ARrange):
                                result_row.append(-1*tarps[i])
                        for i in range(max(ARrange)-len(tarps)): # fit the modelrange_ar-len(ts.ar_poly)):
                            result_row.append(0)
                        for i in range(len(tmaps)):
                            if i < max(MArange):
                                result_row.append(-1*tmaps[i])
                        for i in range(max(MArange)-len(tmaps)):
                            result_row.append(0)

                        # fit the output
                        writer.writerow(result_row)
                        outputFile.flush()

                        counter+=1

def fit_outputs_one_iter(counter, cpmc, ar, ma, armax, mamax):

    result_row = [counter, cpmc, ar, ma]
    # print("{}, {}, {}".format(cpmc, ar, ma))
    ts = ARMATimeSeries(ar,ma, seed=counter) # generate a (ar,ma)-ARMA model
    for i in range(1, len(ts.ar_poly)):
        result_row.append(ts.ar_poly[i])
    for i in range(armax-ar): # fit the modelrange_ar-len(ts.ar_poly)):
        result_row.append(0)
    for i in range(1, len(ts.ma_poly)):
        result_row.append(ts.ma_poly[i])
    for i in range(mamax-ma):
        result_row.append(0)
    tso = get_order(ts.sequence, max_ar=(armax+1), max_ma=(mamax+1)) # use BIC to get order
    result_row.append(tso[0])
    result_row.append(tso[1])
    tarps, tmaps = fit(ts.sequence, tso) # fit the model
    for i in range(len(tarps)):
        if i < armax:
            result_row.append(-1*tarps[i])
    for i in range(armax-len(tarps)): # fit the modelrange_ar-len(ts.ar_poly)):
        result_row.append(0)
    for i in range(len(tmaps)):
        if i < mamax:
            result_row.append(-1*tmaps[i])
    for i in range(mamax-len(tmaps)):
        result_row.append(0)
    # train the HTM
    network = HTM(ts, 5, verbosity=0)
    result_row.append(network.train("rmse", sibt=0, iter_per_cycle=1, normalize_error=True)) # record its error
    ones, res = network.runNetwork()
    tso = get_order(ones, max_ar=(armax+1), max_ma=(mamax+1)) # use BIC to get order
    network.__del__
    del network
    result_row.append(tso[0])
    result_row.append(tso[1])
    tarps, tmaps = fit(ones, tso) # fit the model
    for i in range(len(tarps)):
        if i < armax:
            result_row.append(-1*tarps[i])
    for i in range(armax-len(tarps)): # fit the modelrange_ar-len(ts.ar_poly)):
        result_row.append(0)
    for i in range(len(tmaps)):
        if i < mamax:
            result_row.append(-1*tmaps[i])
    for i in range(mamax-len(tmaps)):
        result_row.append(0)

    return (counter, result_row)

def fit_outputs_unpacker(args):
    return fit_outputs_one_iter(*args)

def fitHTMOutputspar(ARrange, MArange, CPMCrange, n = 10):
    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
    if MArange == []:
        MArange = [0]
    if MArange == [0]:
        _OUTPUT_PATH = "../outputs/TimeSeriesFitter-{}-ARRange({}-{})-MARange-(NULL)-CPMCRange-({}-{}).csv".format(DATE, min(ARrange), max(ARrange), min(CPMCrange), max(CPMCrange))
    else:
        _OUTPUT_PATH = "../outputs/TimeSeriesFitter-ARRange({}-{})-MARange-({}-{})-CPMCRange-({}-{}).csv".format(DATE, min(ARrange), max(ARrange), min(MArange), max(MArange), min(CPMCrange), max(CPMCrange))
    with open(_OUTPUT_PATH, "w") as outputFile:
        writer = csv.writer(outputFile)
        header_row = ["Run #", "CPMC", "AR Order", "MA Order"]
        for ar in ARrange:
            if not ar == 0:
                header_row.append("AR {}-lag Coeff".format(ar))
        for ma in MArange:
            if not ma == 0:
                header_row.append("MA {}-lag Coeff".format(ma))
        header_row.append("BIC AR Order")
        header_row.append("BIC MA Order")
        for ar in ARrange:
            if not ar == 0:
                header_row.append("Fitted AR {}-lag Coeff".format(ar))
        for ma in MArange:
            if not ma == 0:
                header_row.append("Fitted MA {}-lag Coeff".format(ma))
        header_row.append("HTM Error on Prediction")
        header_row.append("HTM Pred AR Order")
        header_row.append("HTM Pred MA Order")
        for ar in ARrange:
            if not ar == 0:
                header_row.append("HTM AR {}-lag Coeff".format(ar))
        for ma in MArange:
            if not ma == 0:
                header_row.append("HTM MA {}-lag Coeff".format(ar))
        writer.writerow(header_row)
        counter = 0

        armax = max(ARrange)
        mamax = max(MArange)

        pool = mp.Pool(processes = n)
        for cpmc in CPMCrange:
            for ar in ARrange:
                for ma in MArange:
                    ids = range(n)
                    ids = [ counter + x for x in ids ]
                    try:
                        results = pool.map(fit_outputs_unpacker, itertools.izip(ids, itertools.repeat(cpmc), itertools.repeat(ar), itertools.repeat(ma), itertools.repeat(armax), itertools.repeat(mamax)))
                        results.sort()
                        results = [r[1] for r in results]

                        for result_row in results:
                            writer.writerow(result_row)
                        counter+=n
                    except ValueError as e:
                        print(e.message)
                        ma-=1 # reset ma
                    except _csv.Error as csve:
                        print(csve)


def main():
    fitHTMOutputspar(range(1,4), range(0), range(2,4), n = 4)

if __name__ == "__main__":
    main()
