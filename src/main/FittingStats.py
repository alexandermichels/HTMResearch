#!/usr/bin/env python
import csv
from time import localtime, strftime
from models.ARMAModels import *

def getDiffs(ARrange, MArange, n = 1000):
    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
    if MArange == []:
        MArange = [0]
    if ARrange == []:
        ARrange = [0]

    print(ARrange)
    print(MArange)
    for ar in ARrange:
        for ma in MArange:
            print(ar,ma)
            _OUTPUT_PATH = "../outputs/FittingDifferences-{}-AR({})-MA({}).csv".format(DATE, ar, ma)
            with open(_OUTPUT_PATH, "w") as outputFile:
                writer = csv.writer(outputFile)
                header_row = ["Run #", "AR Order", "MA Order"]
                last = 0
                for _ar in ARrange:
                    if not _ar == 0:
                        header_row.append("AR {}-lag Coeff".format(_ar))
                    last = _ar
                header_row.append("AR {}-lag Coeff".format(last+1))
                last=0
                for _ma in MArange:
                    if not _ma == 0:
                        header_row.append("MA {}-lag Coeff".format(_ma))
                    last = _ma
                header_row.append("MA {}-lag Coeff".format(last+1))
                header_row.append("BIC AR Order")
                header_row.append("BIC MA Order")
                last = 0
                for _ar in ARrange:
                    if not _ar == 0:
                        header_row.append("Fitted AR {}-lag Coeff".format(_ar))
                    last = _ar
                header_row.append("Fitted AR {}-lag Coeff".format(last+1))
                last=0
                for _ma in MArange:
                    if not ma == 0:
                        header_row.append("Fitted MA {}-lag Coeff".format(_ma))
                    last = _ma
                header_row.append("Fitted MA {}-lag Coeff".format(last+1))
                last = 0
                for _ar in ARrange:
                    if not _ar == 0:
                        header_row.append("Diff in AR {}-lag Coeff".format(_ar))
                    last = _ar
                header_row.append("Diff in AR {}-lag Coeff".format(last+1))
                last=0
                for _ma in MArange:
                    if not _ma == 0:
                        header_row.append("Diff in AR {}-lag Coeff".format(_ma))
                    last = _ma
                header_row.append("Diff in AR {}-lag Coeff".format(last+1))
                writer.writerow(header_row)

                for counter in range(n): # parallelize on n
                    result_row = [counter, ar, ma]
                    ts = ARMATimeSeries(ar,ma) # generate a (ar,ma)-ARMA model
                    for i in range(1, len(ts.ar_poly)):
                        result_row.append(ts.ar_poly[i])
                    for i in range(max(ARrange)-len(ts.ar_poly)+2): # fit the modelrange_ar-len(ts.ar_poly)):
                        result_row.append(0)
                    for i in range(1, len(ts.ma_poly)):
                        result_row.append(ts.ma_poly[i])
                    for i in range(max(MArange)-len(ts.ma_poly)+2):
                        result_row.append(0)
                    tso = get_order(ts.sequence, max_ar=(int(max(ARrange)+1)), max_ma=(int(max(MArange)+1))) # use BIC to get order
                    result_row.append(tso[0])
                    result_row.append(tso[1])
                    tarps, tmaps = fit(ts.sequence, tso) # fit the model
                    if not tarps == None:
                        for i in range(len(tarps)):
                            if i < max(ARrange) +1:
                                result_row.append(tarps[i])
                        for i in range(max(ARrange)-len(tarps)+1): # fit the modelrange_ar-len(ts.ar_poly)):
                            result_row.append(0)
                        for i in range(len(tmaps)):
                            if i < max(MArange)+1:
                                result_row.append(tmaps[i])
                        for i in range(max(MArange)-len(tmaps)+1):
                            result_row.append(0)

                        if len(tarps) >= len(ts.ar_poly[1:]):
                            j = 0
                            for i in range(len(ts.ar_poly)-1):
                                result_row.append(ts.ar_poly[i+1]-tarps[i])
                                j = i
                            for i in range(j, len(tarps)):
                                result_row.append(0-tarps[i])
                            for i in range(max(ARrange)-len(tarps)+1):
                                result_row.append(0)
                        else:
                            j = 0
                            for i in range(len(tarps)):
                                result_row.append(ts.ar_poly[i+1]-tarps[i])
                                j = i
                            j+=1
                            for i in range(j, len(ts.ar_poly)):
                                result_row.append(ts.ar_poly[i])
                            for i in range(max(ARrange)-len(ts.ar_poly)+2):
                                result_row.append(0)
                        if len(tmaps) >= len(ts.ma_poly[1:]):
                            j = 0
                            for i in range(len(ts.ma_poly)-1):
                                result_row.append(ts.ma_poly[i+1]-tmaps[i])
                                j = i
                            for i in range(j, len(tmaps)):
                                result_row.append(0-tmaps[i])
                            for i in range(max(MArange)-len(tmaps)+1):
                                result_row.append(0)
                        else:
                            j = 0
                            for i in range(len(tmaps)):
                                result_row.append(ts.ma_poly[i+1]-tmaps[i])
                                j = i
                            j+=1
                            for i in range(j, len(ts.ma_poly)):
                                result_row.append(ts.ma_poly[i])
                            for i in range(max(MArange)-len(ts.ma_poly)+2):
                                result_row.append(0)

                    else:
                        for i in range(tso[0]):
                            if i < max(ARrange)+1:
                                result_row.append("N/A")
                        for i in range(max(ARrange)-tso[0]+1): # fit the modelrange_ar-len(ts.ar_poly)):
                            result_row.append("N/A")
                        for i in range(tso[1]):
                            if i < max(MArange)+1:
                                result_row.append("N/A")
                        for i in range(max(MArange)-tso[1]+1):
                            result_row.append("N/A")


                        # fit the output
                    writer.writerow(result_row)
                    outputFile.flush()
                outputFile.close()

def main():
    getDiffs(range(1,9), range(0))
    getDiffs(range(0), range(1,9))

if __name__ == "__main__":
    main()
