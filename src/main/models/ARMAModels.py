#!/usr/bin/env python
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima_model import ARMA
from matplotlib import pyplot
import numpy as np

from Sequence import Sequence

class ARMATimeSeries(Sequence):

    def __init__(self, p, q, sigma=1, n=10000):
        self.n = n
        self.p = p
        self.q = q
        self.sigma = sigma
        self.ar_poly = np.r_[1, np.random.rand(p)]
        print("The AR lag polynomial is: {}".format(self.ar_poly))
        self.ma_poly = np.r_[1, np.random.rand(q)]
        print("The MA lag polynomial is: {}".format(self.ma_poly))
        self.sequence = arma_generate_sample(self.ar_poly, self.ma_poly, self.n, self.sigma)
        self.theta = 0

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

def main():
        ts = ARMATimeSeries(2,0)
        for i in range(10):
                print("The time series at {} is {}".format(i, ts.get()))
        ts.plot()

if __name__ == "__main__":
        main()
