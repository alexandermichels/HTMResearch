#!/usr/bin/env python
from matplotlib import pyplot
import numpy as np

class Sequence():

    def __init__(self, n=10000):
        '''
        Parent class for sequences.

        '''
        self.n = n
        self.sequence = np.ones(n)
        self.theta = 0

    def __iter__(self):
        for i in range(self.n):
            yield self.sequence[i]

    def __len__(self):
        return self.n

    def get(self):
        if (self.theta <= self.n):
            self.theta=self.theta+1
            return self.sequence[self.theta-1]

    def get_theta(self):
        return self.theta

    def has_next(self):
        if self.time_left() > 0:
            return True
        return False

    def plot(self):
        # plot
        pyplot.plot(self.sequence, color='red')
        pyplot.show()

    def rewind(self):
        self.set_theta(0)

    def set_theta(self, t):
        if t >= 0:
            self.theta=t
        else:
            #from the end for negatives
            self.theta = self.n + t

    def time_left(self):
        return self.n-self.theta

def main():
    ts = Sequence()
    for i in range(10):
        print("The time series at {} is {}".format(i, ts.get()))
    ts.plot()

if __name__ == "__main__":
    main()
