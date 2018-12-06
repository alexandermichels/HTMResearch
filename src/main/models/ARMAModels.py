#!/usr/bin/env python
from pandas import datetime
from pandas import DataFrame
import time
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima_model import ARMA
from matplotlib import pyplot
import numpy as np
from random import randint
from Sequence import Sequence

class ARMATimeSeries(Sequence):

    def __init__(self, p, q, sigma=1, n=10000):
        self.n = n
        self.p = p
        self.q = q
        np.random.seed(int(time.time()))
        self.sigma = sigma
        self.ar_poly = np.r_[1, np.random.rand(p)]
        print("The AR lag polynomial is: {}".format(self.ar_poly))
        self.ma_poly = np.r_[1, np.random.rand(q)]
        print("The MA lag polynomial is: {}".format(self.ma_poly))
        self.sequence = arma_generate_sample(self.ar_poly, self.ma_poly, self.n, self.sigma)
        self.theta = 0
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

    def __str__(self):
        """
        To string
        """
        return "({},{})-ARMA".format(self.p, self.q)

    def __repr__(self):
        """"
        To string
        """
        return "({},{})-ARMA".format(self.p, self.q)

    def new(self):
        self.ar_poly = np.r_[1, np.random.rand(self.p)]
        print("The AR lag polynomial is: {}".format(self.ar_poly))
        self.ma_poly = np.r_[1, np.random.rand(self.q)]
        print("The MA lag polynomial is: {}".format(self.ma_poly))
        self.sequence = arma_generate_sample(self.ar_poly, self.ma_poly, self.n, self.sigma)
        self.theta = 0
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

    def normalize(self, val=100.0):
        _min = min(self.sequence)
        _max = max(self.sequence)
        furthest = max([abs(_min), abs(_max)])
        self.sequence = [x*val/furthest for x in self.sequence]

def main():
        ts = ARMATimeSeries(3,0)
        for i in range(10):
                print("The time series at {} is {}".format(i, ts.get()))
        ts.plot()

if __name__ == "__main__":
        main()
