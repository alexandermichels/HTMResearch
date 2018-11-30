#!/usr/bin/env python
from matplotlib import pyplot
import numpy as np
from random import randint

class Sequence():

    def __init__(self, n=10000):
        '''
        Parent class for sequences.

        '''
        self.n = n
        self.sequence = np.ones(n)
        self.theta = 0

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
        self._place_in_train_set = self._train_set

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

    def in_eval_set(self):
        if (self.theta >= self._eval_set and self.theta < self._eval_set_end):
            return True
        return False

    def in_test_set(self):
        if (self.theta >= self._test_set and self.theta < self._test_set_end):
            return True
        return False

    def in_train_set(self):
        if (self.theta >= self._train_set and self.theta < self._train_set_end):
            return True
        return False

    def new(self):
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

    def plot(self):
        # plot
        pyplot.plot(self.sequence, color='red')
        pyplot.show()

    def rewind(self):
        self.set_theta(self._train_set)

    def set_theta(self, t):
        if t >= 0:
            self.theta=t
        else:
            #from the end for negatives
            self.theta = self.n + t

    def set_to_eval_theta(self):
        self.theta = self._eval_set

    def set_to_test_theta(self):
        self._place_in_train_set = self.theta
        self.theta = self._test_set

    def set_to_train_theta(self):
        self.theta = self._place_in_train_set + 2
        if (not self.in_train_set()):
            self.theta = self._train_set

    def time_left(self):
        return self.n-self.theta

def main():
    ts = Sequence()
    for i in range(10):
        print("The time series at {} is {}".format(i, ts.get()))
    ts.plot()

if __name__ == "__main__":
    main()
