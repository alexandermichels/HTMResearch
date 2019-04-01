#!/usr/bin/env python
import csv
from time import localtime, strftime
from models.ARMAModels import *

from HTM import *

def getDiffs(ARrange, MArange, n = 1000):
    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
    if MArange == []:
        MArange = [0]
    if ARrange == []:
        ARrange = [0]

    EXTRA_TERMS = 2

    print(ARrange)
    print(MArange)
    for ar in ARrange:
        for ma in MArange:
            '''
            For a particular p, q in ARrange x MArange
            '''

            _OUTPUT_PATH = "../outputs/FittingDifferences-{}-AR({})-MA({}).csv".format(DATE, ar, ma)
            with open(_OUTPUT_PATH, "w") as outputFile:
                writer = csv.writer(outputFile)
                header_row = ["Run #", "AR Order", "MA Order"]
                last = 0
                if not ar == 0:
                    for _ar in range(1,ar+1):
                        header_row.append("AR {}-lag Coeff".format(_ar))
                if not ma == 0:
                    for _ma in range(1,ma+1):
                        header_row.append("MA {}-lag Coeff".format(_ma))
                header_row.append("BIC AR Order")
                header_row.append("BIC MA Order")

                for _ar in range(1, ar+EXTRA_TERMS+1):
                    header_row.append("Fitted AR {}-lag Coeff".format(_ar))
                for _ma in range(1, ma+EXTRA_TERMS+1):
                    header_row.append("Fitted MA {}-lag Coeff".format(_ma))
                for _ar in range(1, ar+EXTRA_TERMS+1):
                    header_row.append("Diff in AR {}-lag Coeff".format(_ar))
                for _ma in range(1, ma+EXTRA_TERMS+1):
                    header_row.append("Diff in MA {}-lag Coeff".format(_ma))
                writer.writerow(header_row)

                for counter in range(n): # parallelize on n
                    result_row = [counter, ar, ma]
                    ts = ARMATimeSeries(ar,ma) # generate a (ar,ma)-ARMA model
                    for i in range(1, len(ts.ar_poly)):
                        result_row.append(ts.ar_poly[i])
                    for i in range(1, len(ts.ma_poly)):
                        result_row.append(ts.ma_poly[i])

                    tso = get_order(ts.sequence, max_ar=(int(ar+2)), max_ma=(int(ma+2))) # use BIC to get order
                    result_row.append(tso[0])
                    result_row.append(tso[1])
                    tarps, tmaps = fit(ts.sequence, tso) # fit the model

                    if not tarps == None:
                        for i in range(len(tarps)):
                            if i < ar +EXTRA_TERMS:
                                result_row.append(tarps[i])
                        for i in range(ar-len(tarps)+EXTRA_TERMS): # fit the modelrange_ar-len(ts.ar_poly)):
                            result_row.append(0)

                        for i in range(len(tmaps)):
                            if i < ma+EXTRA_TERMS:
                                result_row.append(tmaps[i])
                        for i in range(ma-len(tmaps)+EXTRA_TERMS):
                            result_row.append(0)

                        '''
                        Diffs of AR terms
                        '''
                        minlen = min(len(tarps),len(ts.ar_poly[1:]))
                        for i in range(minlen):
                            result_row.append(ts.ar_poly[i+1]+tarps[i])

                        if len(tarps) >= len(ts.ar_poly[1:]):
                            for i in range(ar, len(tarps)):
                                if i < ar+EXTRA_TERMS:
                                    result_row.append(0+tarps[i])
                        else:
                            for i in range(len(tarps), ar):
                                if i < ar+EXTRA_TERMS:
                                    result_row.append(ts.ar_poly[i+1])

                        maxlen = max(ar, len(tarps))
                        for i in range(ar-maxlen+EXTRA_TERMS):
                            result_row.append(0)

                        '''
                        Diffs of MA terms
                        '''
                        minlen = min(len(tmaps),len(ts.ma_poly[1:]))
                        for i in range(minlen):
                            result_row.append(ts.ma_poly[i+1]-tmaps[i])

                        if len(tmaps) >= len(ts.ma_poly[1:]):
                            for i in range(ma, len(tmaps)):
                                if i < ma+EXTRA_TERMS:
                                    result_row.append(0-tmaps[i])
                        else:
                            for i in range(len(tmaps), ma):
                                if i < ma+EXTRA_TERMS:
                                    result_row.append(ts.ma_poly[i+1])
                        maxlen = max(ma, len(tmaps))
                        for i in range(ma-maxlen+EXTRA_TERMS):
                            result_row.append(0)

                    else:
                        for i in range(ar+EXTRA_TERMS): # fit the modelrange_ar-len(ts.ar_poly)):
                            result_row.append("N/A")
                        for i in range(ma+EXTRA_TERMS):
                            result_row.append("N/A")


                        # fit the output
                    writer.writerow(result_row)
                    outputFile.flush()
                outputFile.close()

def singleModelGetDiffs(model, n, EXTRA_TERMS=2):
    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
    _OUTPUT_PATH = "../outputs/SingleModelFittingDifferences-{}.csv".format(DATE, model)
    with open(_OUTPUT_PATH, "w") as outputFile:
        writer = csv.writer(outputFile)

        header_row = ["Run #", "AR Order", "MA Order"]
        if not model.p == 0:
            for _ar in range(1,model.p+1):
                header_row.append("AR {}-lag Coeff".format(_ar))
        if not model.q == 0:
            for _ma in range(1,model.q+1):
                header_row.append("MA {}-lag Coeff".format(_ma))
        header_row.append("BIC AR Order")
        header_row.append("BIC MA Order")
        for _ar in range(1,model.p+EXTRA_TERMS+1):
            header_row.append("Fitted AR {}-lag Coeff".format(_ar))
        for _ma in range(1,model.q+EXTRA_TERMS+1):
            header_row.append("Fitted MA {}-lag Coeff".format(_ma))
        for _ar in range(1,model.p+EXTRA_TERMS+1):
            header_row.append("Diff in AR {}-lag Coeff".format(_ar))
        for _ma in range(1,model.q+EXTRA_TERMS+1):
            header_row.append("Diff in MA {}-lag Coeff".format(_ma))
        writer.writerow(header_row)

        for counter in range(n): # parallelize on n
            result_row = [counter, model.p, model.q]
            model.new(newPoly=False)

            for i in range(1, len(model.ar_poly)):
                result_row.append(model.ar_poly[i])
            for i in range(1, len(model.ma_poly)):
                result_row.append(model.ma_poly[i])
            tso = get_order(model.sequence, max_ar=(model.p+2), max_ma=(model.q+2)) # use BIC to get order
            result_row.append(tso[0])
            result_row.append(tso[1])
            tarps, tmaps = fit(model.sequence, tso) # fit the model

            if not tarps == None:
                for i in range(len(tarps)):
                    if i < model.p +EXTRA_TERMS:
                        result_row.append(tarps[i])
                for i in range(model.p-len(tarps)+EXTRA_TERMS): # fit the modelrange_ar-len(ts.ar_poly)):
                    result_row.append(0)

                for i in range(len(tmaps)):
                    if i < model.q+EXTRA_TERMS:
                        result_row.append(tmaps[i])
                for i in range(model.q-len(tmaps)+EXTRA_TERMS):
                    result_row.append(0)

                '''
                Diffs of AR terms
                '''
                minlen = min(len(tarps),len(model.ar_poly[1:]))
                for i in range(minlen):
                    result_row.append(model.ar_poly[i+1]+tarps[i])

                if len(tarps) >= len(model.ar_poly[1:]):
                    for i in range(model.p, len(tarps)):
                        if i < model.p+EXTRA_TERMS:
                            result_row.append(0+tarps[i])
                else:
                    for i in range(len(tarps), model.p):
                        if i < model.p+EXTRA_TERMS:
                            result_row.append(model.ar_poly[i+1])

                maxlen = max(model.p, len(tarps))
                for i in range(model.p-maxlen+EXTRA_TERMS):
                    result_row.append(0)

                '''
                Diffs of MA terms
                '''
                minlen = min(len(tmaps),len(model.ma_poly[1:]))
                for i in range(minlen):
                    result_row.append(model.ma_poly[i+1]-tmaps[i])

                if len(tmaps) >= len(model.ma_poly[1:]):
                    for i in range(model.q, len(tmaps)):
                        if i < model.q+EXTRA_TERMS:
                            result_row.append(0-tmaps[i])
                else:
                    for i in range(len(tmaps), model.q):
                        if i < model.q+EXTRA_TERMS:
                            result_row.append(model.ma_poly[i+1])

                maxlen = max(model.q, len(tmaps))
                for i in range(model.q-maxlen+EXTRA_TERMS):
                    result_row.append(0)
            else:
                for i in range(model.p+EXTRA_TERMS):
                    result_row.append("N/A")
                for i in range(model.q+EXTRA_TERMS):
                    result_row.append("N/A")
            writer.writerow(result_row)
            outputFile.flush()
        outputFile.close()

def train_HTM_on_model(model):
    network = HTM(model, 5, verbosity=0)
    network.train("rmse", sibt=0, iter_per_cycle=1, normalize_error=True)
    ones, res = network.runNetwork()
    return ones

def test_HTM_output(arr, ar_max, ma_max):
    tso = get_order(arr, max_ar=ar_max, max_ma=ma_max)
    print(tso)
    tarps, tmaps = fit(arr, tso)
    print(tarps)
    print(tmaps)

def main():
    polys = [[1, 0, 0, 0, .9], [1, 0, 0, 0, 0, 0, 0, 0, .9], [1, 0, .2, .8], [1, 0, .5, 0, 0, .5]]
    for poly in polys:
        model = ARMATimeSeries( len(poly)-1, 0, 1, ar_poly = poly)  # p, q, sigma=1, n=1000, normalize=True, seed=int(time.time()), ar_poly = None, ma_poly = None)
        singleModelGetDiffs(model, 1500)
        model = ARMATimeSeries( 0, len(poly)-1, 1, ma_poly = poly)  # p, q, sigma=1, n=1000, normalize=True, seed=int(time.time()), ar_poly = None, ma_poly = None)
        singleModelGetDiffs(model, 1500)
    '''model = ARMATimeSeries( 6, 0, 1, ar_poly = [1, 0, 0, .4, 0, .3, .3])  # p, q, sigma=1, n=1000, normalize=True, seed=int(time.time()), ar_poly = None, ma_poly = None)
    singleModelGetDiffs(model, 1500)
    model = ARMATimeSeries( 0, 6, 1, ma_poly = [1, 0, 0, .4, 0, .3, .3])  # p, q, sigma=1, n=1000, normalize=True, seed=int(time.time()), ar_poly = None, ma_poly = None)
    test_HTM_output(train_HTM_on_model(model), 2,len(poly)+2)'''
    # getDiffs(range(1,9), range(0))
    # getDiffs(range(0), range(1,9))

if __name__ == "__main__":
    main()
