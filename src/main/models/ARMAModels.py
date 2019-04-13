#!/usr/bin/env python
from pandas import datetime
from pandas import DataFrame
import csv, time
from datetime import datetime
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import arma_order_select_ic
from matplotlib import pyplot
import numpy as np
from random import randint, uniform
from Sequence import Sequence
from math import sqrt
np.warnings.filterwarnings('ignore')
"""Handles Ctrl+C"""
def signal_handler(sig, frame):
        sys.exit(0)

class ARMATimeSeries(Sequence):

    def __init__(self, p, q, sigma=1, n=1000, normalize=True, seed=None, ar_poly = None, ma_poly = None):
        if seed == None:
            dt = datetime.now()
            seed = int(time.time()) * 100000 + dt.microsecond
            seed = int(seed) % 4294967295 # 2^32 - 1
        self.n = n
        self.p = p
        self.q = q
        self.norm=normalize
        np.random.seed(seed)
        self.sigma = sigma
        if ar_poly == None:
            self.ar_poly = [1]
            sum = 0
            if self.p > 0:
                for i in range(p):
                    temp = randint(0,1000)
                    self.ar_poly.append(temp)
                    sum+=temp
                if sum > 0:
                    norm = uniform(0,1)/sum
                    for i in range(1,p+1):
                        self.ar_poly[i]=self.ar_poly[i]*norm
        else:
            self.ar_poly = ar_poly
            self.p = len(self.ar_poly)-1
        if ma_poly == None:
            self.ma_poly = [1]
            sum = 0
            if self.q > 0:
                for i in range(q):
                    temp = randint(0,1000)
                    self.ma_poly.append(temp)
                    sum+=temp
                if sum > 0:
                    norm = uniform(0,1)/sum
                    for i in range(1,q+1):
                        self.ma_poly[i]=self.ma_poly[i]*norm
        else:
            self.ma_poly = ma_poly
            self.q = len(ma_poly)-1

        self.sequence = arma_generate_sample(self.ar_poly, self.ma_poly, self.n, self.sigma)
        self.theta = 0
        if self.norm:
            self.normalize()

        if randint(0,1) == 0:
            if randint(0,1) == 0:
                self._test_set = 0
                self._test_set_end = int(1*self.n/10)
                self._eval_set = int(1*self.n/10)
                self._eval_set_end = int(2*self.n/10)
            else:
                self._eval_set = 0
                self._eval_set_end = int(1*self.n/10)
                self._test_set = int(1*self.n/10)
                self._test_set_end = int(2*self.n/10)
            self._train_set = int(2*self.n/10)+1
            self._train_set_end = self.n
        else:
            if randint(0,1) == 0:
                self._test_set = int(8*self.n/10)
                self._test_set_end = int(9*self.n/10)
                self._eval_set = int(9*self.n/10)
                self._eval_set_end = self.n
            else:
                self._eval_set = int(8*self.n/10)
                self._eval_set_end = int(9*self.n/10)
                self._test_set = int(9*self.n/10)
                self._test_set_end = self.n
            self._train_set = 0
            self._train_set_end = int(8*self.n/10)
        self._place_in_train_set = self._train_set

        self.sequence = np.array(self.sequence)

    def __str__(self):
        """
        To string
        """
        return "({},{})-ARMA".format(self.p, self.q)

    def __repr__(self):
        """"
        To string
        """
        return self.__str__()

    def fit(self, result = "armaparams"):
        arma = ARMA(self.sequence, (self.p, self.q))
        try:
            res = arma.fit(trend="nc", maxiter=len(self.sequence), disp=-1)
        except Exception as e1:
            print(e1)
            try:
                print("LBFGS failed for {}, trying BFGS".format(self.__str__))
                res = arma.fit(trend="nc", solver="bfgs", maxiter=len(self.sequence), disp=-1)
            except Exception as e2:
                print(e2)
                try:
                    print("BFGS failed for {}, trying netwon".format(self.__str__))
                    res = arma.fit(trend="nc", solver="newton", maxiter=len(self.sequence), disp=-1)
                except Exception as e3:
                    print(e3)
                    try:
                        print("Newton failed for {}, trying Nelder-Mead".format(self.__str__))
                        res = arma.fit(trend="nc", solver="nm", maxiter=len(self.sequence), disp=-1)
                    except Exception as e4:
                        print(e4)
                        try:
                            print("Nelder-Mead failed for {}, trying Conjugate Gradient".format(self.__str__))
                            res = arma.fit(trend="nc", solver="cg", maxiter=len(self.sequence), disp=-1)
                        except Exception as e5:
                            print(e5)
                            try:
                                print("Conjugate Gradient failed for {}, trying Non-Conjugate Gradient".format(self.__str__))
                                res = arma.fit(trend="nc", solver="ncg", maxiter=len(self.sequence), disp=-1)
                            except Exception as e6:
                                print(e6)
                                try:
                                    print("Non-Conjugate Gradient failed for {}, trying Powell".format(self.__str__))
                                    res = arma.fit(trend="nc", solver="powell", maxiter=len(self.sequence), disp=-1)
                                except:
                                    return None, None
        if result == "armaparams":
            return list(res.arparams), list(res.maparams)
        elif result == "sigma2":
            return res.sigma2

    def new(self, newPoly = True):
        if newPoly:
            self.ar_poly = np.r_[1, np.random.rand(self.p)]
            print("The AR lag polynomial is: {}".format(self.ar_poly))
            self.ma_poly = np.r_[1, np.random.rand(self.q)]
            print("The MA lag polynomial is: {}".format(self.ma_poly))
        else:
            print("The AR lag polynomial is: {}".format(self.ar_poly))
            print("The MA lag polynomial is: {}".format(self.ma_poly))
        self.sequence = arma_generate_sample(self.ar_poly, self.ma_poly, self.n, self.sigma)
        self.theta = 0
        if self.norm:
            self.normalize()

        self._train_set = -1
        self._train_set_end = -1
        self._test_set = -1
        self._test_set_end = -1
        self._eval_set = -1
        self._eval_set_end = -1

        if randint(0,1) == 0:
            if randint(0,1) == 0:
                self._test_set = 0
                self._test_set_end = int(1*self.n/10)
                self._eval_set = int(1*self.n/10)
                self._eval_set_end = int(2*self.n/10)
            else:
                self._eval_set = 0
                self._eval_set_end = int(1*self.n/10)
                self._test_set = int(1*self.n/10)
                self._test_set_end = int(2*self.n/10)
            self._train_set = int(2*self.n/10)+1
            self._train_set_end = self.n
        else:
            if randint(0,1) == 0:
                self._test_set = int(8*self.n/10)
                self._test_set_end = int(9*self.n/10)
                self._eval_set = int(9*self.n/10)
                self._eval_set_end = self.n
            else:
                self._eval_set = int(8*self.n/10)
                self._eval_set_end = int(9*self.n/10)
                self._test_set = int(9*self.n/10)
                self._test_set_end = self.n
            self._train_set = 0
            self._train_set_end = int(8*self.n/10)

        self.sequence = np.array(self.sequence)

    def normalize(self, val=100.0):
        _min = min(self.sequence)
        _max = max(self.sequence)
        furthest = max([abs(_min), abs(_max)])
        self.sequence = [x*val/furthest for x in self.sequence]

    def plot(self):
        pyplot.plot(self.sequence, color='blue', label="series")
        pyplot.legend(loc="lower right")
        pyplot.autoscale(enable=True, axis='x', tight=True)
        pyplot.show()

def get_order(arr, max_ar=4, max_ma=2):
    return arma_order_select_ic(arr, max_ar=max_ar, max_ma=max_ma, trend="nc").bic_min_order

def fit(arr, order):
    try:
        res = ARMA(arr, order).fit(trend="nc", maxiter=len(arr), disp=-1)
    except Exception as e1:
        print(e1)
        try:
            print("LBFGS failed for {}, trying BFGS".format(arr))
            res = ARMA(arr, order).fit(trend="nc", solver="bfgs", maxiter=len(arr), disp=-1)
        except Exception as e2:
            print(e2)
            try:
                print("BFGS failed for {}, trying netwon".format(arr))
                res = ARMA(arr, order).fit(trend="nc", solver="newton", maxiter=len(arr), disp=-1)
            except Exception as e3:
                print(e3)
                try:
                    print("Newton failed for {}, trying Nelder-Mead".format(arr))
                    res = ARMA(arr, order).fit(trend="nc", solver="nm", maxiter=len(arr), disp=-1)
                except Exception as e4:
                    print(e4)
                    try:
                        print("Nelder-Mead failed for {}, trying Conjugate Gradient".format(arr))
                        res = ARMA(arr, order).fit(trend="nc", solver="cg", maxiter=len(arr), disp=-1)
                    except Exception as e5:
                        print(e5)
                        try:
                            print("Conjugate Gradient failed for {}, trying Non-Conjugate Gradient".format(arr))
                            res = ARMA(arr, order).fit(trend="nc", solver="ncg", maxiter=len(arr), disp=-1)
                        except Exception as e6:
                            print(e6)
                            try:
                                print("Non-Conjugate Gradient failed for {}, trying Powell".format(arr))
                                res = ARMA(arr, order).fit(trend="nc", solver="powell", maxiter=len(arr), disp=-1)
                            except:
                                return None, None
    return list(res.arparams), list(res.maparams)

def main():
    ts = ARMATimeSeries(6,0, sigma=1, ar_poly=[1,0,0,.4, 0, .3, .3], normalize=False)
    ts.plot()

if __name__ == "__main__":
        main()
