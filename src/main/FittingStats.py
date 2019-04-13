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
    _OUTPUT_PATH = "../outputs/SingleModelFittingDifferences-{}-{}.csv".format(DATE, model)
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

def train_HTM_on_model(model, params, htmparams=None):
    network = HTM(model, params["rdse"], params=params, verbosity=3)
    network.train("rmse", sibt=params["sibt"], iter_per_cycle=params["iter_per_cycle"], weights=params["weights"], normalize_error=True)
    ones, res = network.runNetwork(learning=False)
    pyplot.plot(model.sequence, color='red', label="series")
    pyplot.plot(ones, color='blue', label="predictions")
    pyplot.legend(loc="lower right")
    pyplot.autoscale(enable=True, axis='x', tight=True)
    pyplot.show()
    return ones

def test_HTM_output(arr, ar_max, ma_max):
    tso = get_order(arr, max_ar=ar_max, max_ma=ma_max)
    print(tso)
    tarps, tmaps = fit(arr, tso)
    if not tarps == None:
        tarps = [-1*x for x in tarps]
        print(tarps)
        print(tmaps)

def fit_models_of_interest():
    polys = [[1, 0, 0, 0, .8], [1, 0, .2, .8], [1, 0, .5, 0, 0, .5], [1, 0, 0, 0, 0, 0, 0, 0, .9]]
    for poly in polys:
        model = ARMATimeSeries( len(poly)-1, 0, 1, ar_poly = poly)  # p, q, sigma=1, n=1000, normalize=True, seed=int(time.time()), ar_poly = None, ma_poly = None)
        singleModelGetDiffs(model, 1500)
        model = ARMATimeSeries( 0, len(poly)-1, 1, ma_poly = poly)  # p, q, sigma=1, n=1000, normalize=True, seed=int(time.time()), ar_poly = None, ma_poly = None)
        singleModelGetDiffs(model, 1500)

def test_HTM_model_of_interest(selector):
    if selector == "AR3-Lite":
        model = ARMATimeSeries(3, 0, 1, ar_poly = [1, 0, 0.2, 0.8])
        test_HTM_output(train_HTM_on_model(model, params= { "rdse": 4.05939793188752, "sibt": 36, "iter_per_cycle": 1, "weights" : {1: 1.0, 2: 3.950446994867, 3: 0, 4: 7.58486758256058, 5: 8.14097282846868, 6: 4.66048066863582, 7: 6.15695358785945, 8: 6.68821590225303, 9: 10 }}), len(model.ar_poly)+1, len(model.ma_poly)+1)
    elif selector == "AR3-Full":
        param_dict = { "spParams" : { "potentialPct": 0.00005, "numActiveColumnsPerInhArea": 44, "synPermConnected": 0.00005, "synPermInactiveDec": 0.00005 }, "tmParams" : { "activationThreshold": 11, "newSynapseCount" : 15 }}
        model = ARMATimeSeries(3, 0, 1, ar_poly = [1, 0, 0.2, 0.8])
        test_HTM_output(train_HTM_on_model(model, params= { "rdse": 3, "sibt": 22, "iter_per_cycle": 1, "weights": {1: 1.0, 2: 1.99406959896973, 3: 10, 4: 0, 5: 10, 6: 9.54327597693671, 7: 0, 8: 0, 9: 0 }}, htmparams=param_dict), len(model.ar_poly)+1, len(model.ma_poly)+1)
    elif selector == "AR4-Lite":
        model = ARMATimeSeries(4, 0, 1, ar_poly = [1, 0, 0, 0, 0.8], seed=12345)  # p, q, sigma=1, n=1000, normalize=True, seed=int(time.time()), ar_poly = None, ma_poly = None)
        test_HTM_output(train_HTM_on_model(model, params= { "rdse": 10, "sibt": 0, "iter_per_cycle": 1, "weights" : {1: 1.0, 2: 8.64405184526634, 3: 10, 4: 10, 5: 0, 6: 10, 7: 6.02510913831131, 8: 0, 9: 0 }}), len(model.ar_poly)+1, len(model.ma_poly)+1)
    elif selector == "AR4-Full":
        param_dict = { "spParams" : { "potentialPct": 0.00005, "numActiveColumnsPerInhArea": 48, "synPermConnected": 0.147007617546864, "synPermInactiveDec": 0.048096924657991}, "tmParams" : { "activationThreshold": 30, "newSynapseCount" : 31 }}
        model = ARMATimeSeries(4, 0, 1, ar_poly = [1, 0, 0, 0, 0.8])
        test_HTM_output(train_HTM_on_model(model, params= { "rdse": 9.31569877677968, "sibt": 9, "iter_per_cycle": 2, "weights": {1: 1.0, 2: 3.39009564264216, 3: 2.48358343152521, 4: 2.7612182073368, 5: 3.71140062541657, 6: 8.55307831696238, 7: 0.439759989458677, 8: 8.91798126584945, 9: 1.81781701185768 }}, htmparams=param_dict), len(model.ar_poly)+1, len(model.ma_poly)+1)
    elif selector == "AR5-Lite":
        model = ARMATimeSeries(5, 0, 1, ar_poly = [1, 0, .5, 0, 0, .5])
        test_HTM_output(train_HTM_on_model(model, params= { "rdse": 5.78082666538634, "sibt": 31, "iter_per_cycle": 1, "weights" : {1: 1.0, 2: 2.09203073084044, 3: 9.47628646177412, 4: 5.97732661225504, 5: 3.16532406316063, 6: 2.33171457558926, 7: 1.8498652765416, 8: 9.21281900242032, 9: 4.74165300420722}}), len(model.ar_poly)+1, len(model.ma_poly)+1)
    elif selector == "AR6-Lite":
        model = ARMATimeSeries(6, 0, 1, ar_poly = [1, 0, 0, .4, 0, .3, .3])  # p, q, sigma=1, n=1000, normalize=True, seed=int(time.time()), ar_poly = None, ma_poly = None)
        test_HTM_output(train_HTM_on_model(model, params= { "rdse": 3.850623926016, "sibt": 4, "iter_per_cycle": 1, "weights": {1: 1.0, 2: 0.806763834261459, 3: 10, 4: 1.30456218335413, 5: 0, 6: 10, 7: 10, 8: 0, 9: 0 }}), len(model.ar_poly)+1, len(model.ma_poly)+1)
    elif selector == "AR6-Full":
        param_dict = { "spParams" : { "potentialPct": 0.21997701642794, "numActiveColumnsPerInhArea": 68, "synPermConnected": .00001, "synPermInactiveDec": 0.1}, "tmParams" : { "activationThreshold": 8, "newSynapseCount" : 15 }}
        model = ARMATimeSeries(6, 0, 1, ar_poly = [1, 0, 0, .4, 0, .3, .3])  # p, q, sigma=1, n=1000, normalize=True, seed=int(time.time()), ar_poly = None, ma_poly = None)
        test_HTM_output(train_HTM_on_model(model, params= { "rdse": 6.00957603768008, "sibt": 6, "iter_per_cycle": 2, "weights": {1: 1.0, 2: 6.35637957153978, 3: 2.41779944347232, 4: 1.98962361570088, 5: 4.19278364734038, 6: 1.48386333867527, 7: 8.17271714154883, 8: 10, 9: 0.263161870966194 }}, htmparams=param_dict), len(model.ar_poly)+1, len(model.ma_poly)+1)
    elif selector=="MA6-Full":
        param_dict = { "spParams" : { "potentialPct": 0.21997701642794, "numActiveColumnsPerInhArea": 68, "synPermConnected": .00005, "synPermInactiveDec": 0.1 }, "tmParams" : { "activationThreshold": 8, "newSynapseCount" : 15 }}
        model = ARMATimeSeries(0, 6, 1, ma_poly = [1, 0, 0, .4, 0, .3, .3])  # p, q, sigma=1, n=1000, normalize=True, seed=int(time.time()), ar_poly = None, ma_poly = None)
        test_HTM_output(train_HTM_on_model(model, params= { "rdse": 10, "sibt": 50, "iter_per_cycle": 1, "weights": {1: 1.0, 2: 3.24045225732635, 3: 10, 4: 0, 5: 0, 6: 10, 7: 0, 8: 10, 9: 0 }}, htmparams=param_dict), len(model.ar_poly)+1, len(model.ma_poly)+1)
    elif selector == "Shaffer":
        model = ARMATimeSeries(6, 0, 1, n =10000,  ar_poly = [1, 0, 0, -.4, 0, -.3, -.3])  # p, q, sigma=1, n=1000, normalize=True, seed=int(time.time()), ar_poly = None, ma_poly = None)
        model.plot()
        test_HTM_output(train_HTM_on_model(model, params= { "rdse": 3.850623926016, "sibt": 4, "iter_per_cycle": 1, "weights": {1: 1.0, 2: 0.806763834261459, 3: 10, 4: 1.30456218335413, 5: 0, 6: 10, 7: 10, 8: 0, 9: 0 }}), len(model.ar_poly)+1, len(model.ma_poly)+1)

def test_HTM_models_of_interest():
    selectors = [ "AR3-Lite", "AR3-Full", "AR4-Lite", "AR4-Full", "AR5-Lite", "AR6-Lite", "AR6-Full", "MA6-Full" ]
    for selector in selectors:
        print("Testing {}....".format(selector))
        test_HTM_model_of_interest(selector)
        print("\n\n\n")

def compare_fitted_rmse(seeded, n):
    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
    _OUTPUT_PATH = "../outputs/CompareFittedRMSE-{}.csv".format(DATE)
    with open(_OUTPUT_PATH, "w") as outputFile:
        writer = csv.writer(outputFile)
        for theMovieHerIsUnderratedAndILoveJoaquinPhoenix in range(n):
            ts = ARMATimeSeries(4,0, sigma=0.00000000001, ar_poly=[1,0,0,0,.8], seed=12345)
            bic = get_order(ts.sequence, 4, 2)
            ar_poly, ma_poly = fit(ts.sequence, bic)
            ar_poly = [-1*x for x in ar_poly] # ar coeff come out negative
            ar_poly, ma_poly = [1]+ar_poly, [1]+ma_poly
            print(ar_poly)
            print(ma_poly)
            if seeded:
                modelofinstance = ARMATimeSeries(bic[0], bic[1], sigma=0.00000000001, ar_poly=ar_poly, ma_poly=ma_poly, seed=12345)
            else:
                modelofinstance = ARMATimeSeries(bic[0], bic[1], sigma=0.00000000001, ar_poly=ar_poly, ma_poly=ma_poly)
            ts.new(False)
            rmse = 0
            for i in range(len(modelofinstance)):
                rmse+=sqrt((ts.get()-modelofinstance.get())**2)
            _range = max(ts.get_range(), modelofinstance.get_range())
            rmse = rmse/_range/len(modelofinstance)
            print(rmse)
            writer.writerow([rmse])
        outputFile.close()


def main():
    test_HTM_model_of_interest("Shaffer")

if __name__ == "__main__":
    main()
