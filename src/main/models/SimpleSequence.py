from matplotlib import pyplot
import numpy as np
from random import randint, random

from Sequence import Sequence

class OneTermSimpleSequence(Sequence):

    def __init__(self, order, lag, determinism=1, n=10000):
        """
        Creates a sequence object with a lag-ary operation defining transitions from each othe order elements to the other order.

        :param order: the number of elements that may appear in the sequence.
        :param lags: a binary array
        """
        self.theta = 0
        self.order = order
        self.lag = lag
        self.determinism = determinism
        self.n = n
        self.sequence = np.ones(n)

        #make the transition matrix
        self.transition = np.ones(order)
        goodMatrix = False
        while not goodMatrix:
            #make a matrix
            for i in range(len(self.transition)):
                if random() < self.determinism:
                    self.transition[i] = randint(1, self.order)
                else:
                    self.transition[i] = 0

            #check the matrix
            goodMatrix = True
            for i in range(len(self.transition)):
                if self.transition[i] == i +1:
                    print("Rejected matrix for having one cycle")
                    goodMatrix = False

        #seed the sequence
        counter = 0
        while counter < self.lag:
            self.sequence[counter] = randint(1, self.order)
            counter+=1
        #get rid of the fixed points
        while counter < self.n:
            where_to = int(self.sequence[counter-self.lag]-1)
            print(where_to)
            next = self.transition[where_to]
            if next == 0:
                self.sequence[counter] = randint(1, self.order)
            else:
                self.sequence[counter] = next
            counter+=1



def main():
    ts = OneTermSimpleSequence(7,3)
    for i in range(100):
        print("The time series at {} is {}".format(i, ts.get()))
    ts.plot()

if __name__ == "__main__":
    main()
