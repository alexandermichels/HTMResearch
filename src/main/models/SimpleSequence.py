from matplotlib import pyplot
import numpy as np

from Sequence import Sequence

class SimpleSequence(Sequence):

    def __init__(self, order, lags, determinism=1.0, n=10000):
        """
        Creates a sequence object with a lag-ary operation defining transitions from each othe order elements to the other order.

        :param order: the number of elements that may appear in the sequence.
        :param lags: a binary array
        """
        self.n = n
        self.theta = 0
        self.order = order
        self.dim = np.sum(lags, dtype=np.int32)
        


def main():
    ts = SimpleSequence()
    for i in range(10):
        print("The time series at {} is {}".format(i, ts.get()))
    ts.plot()

if __name__ == "__main__":
    main()
