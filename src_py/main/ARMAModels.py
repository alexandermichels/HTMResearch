from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima_model import ARMA
from matplotlib import pyplot
import numpy as np

class ARMATimeSeries():

    def __init__(self, p, q, sigma=.5, n=10000):
        self.n = n
        self.p = p
        self.q = q
        self.ar_poly = np.r_[1, np.random.rand(p)]
        print("The AR lag polynomial is: {}".format(self.ar_poly))
        self.ma_poly = np.r_[1, np.random.rand(q)]
        print("The MA lag polynomial is: {}".format(self.ma_poly))
        self.time_series = arma_generate_sample(self.ar_poly, self.ma_poly, n, sigma)
        self.theta = 0

    def __iter__(self):
        for i in range(self.n):
            yield self.time_series[i]

    def get(self):
        if (self.theta <= self.n):
            self.theta=self.theta+1
            return self.time_series[self.theta-1]

    def get_theta(self):
        return self.theta

    def has_next(self):
        if self.time_left() > 0:
            return True
        return False

    def plot(self):
        # plot
        pyplot.plot(self.time_series, color='red')
        pyplot.show()

    def set_theta(self, t):
        if theta >= 0:
            self.theta=t
        else:
            #from the end for negatives
            self.theta = self.n + t


    def time_left(self):
        return self.n-self.theta

def main():
    ts = ARMATimeSeries(2,0)
    for i in range(10):
        print("The time series at {} is {}".format(i, ts.get()))
    ts.plot()

if __name__ == "__main__":
    main()
