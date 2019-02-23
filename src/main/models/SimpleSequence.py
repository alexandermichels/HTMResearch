from matplotlib import pyplot
import numpy as np
from random import randint, random

from Sequence import Sequence


class VeryBasicSequence(Sequence):

    def __init__(self, pattern=2, n=10000):
        """
        Creates a very basic sequence which is just the following two sequences repeated:

        1212
        """
        self.theta = 0
        self.n = int((n+3)/4)*4 # makes n the smallest multiple of four bigger than the required number
        self.sequence = np.ones(n)
        self.pattern = pattern

        counter = 0
        # make the very basic sequence
        if self.pattern == 1:
            while counter < self.n:
                if counter % 2 == 0:
                    self.sequence[counter] = 1
                else:
                    self.sequence[counter] = 2
                counter+=1
        elif self.pattern == 2:
            while counter < self.n:
                if counter % 4 == 0:
                    self.sequence[counter] = 1
                elif counter % 4 == 2:
                    self.sequence[counter] = 3
                else:
                    self.sequence[counter] = 2
                counter+=1
        elif self.pattern == 3:
            while counter < self.n:
                if counter % 2 == 0:
                    self.sequence[counter] = 1
                else:
                    self.sequence[counter] = 3
            counter+=1
        elif self.pattern == 4:
            while counter < self.n:
                if counter % 2 == 0:
                    self.sequence[counter] = 78
                else:
                    self.sequence[counter] = 79
                counter+=1
        elif self.pattern == 5:
            while counter < self.n:
                if counter % 8 == 0:
                    self.sequence[counter] = 1
                elif counter % 8 == 3:
                    self.sequence[counter] = 3
                elif counter % 8 == 4:
                    self.sequence[counter] = 5
                elif counter % 8 == 7:
                    self.sequence[counter] = 4
                else:
                    self.sequence[counter] = 2


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

    def __getitem__(self, i):
        return self.sequence[i]

    def __str__(self):
        """
        To string
        """
        if self.pattern == 1:
            patt = "1212"
        elif self.pattern == 2:
            patt = "1232"
        elif self.pattern == 3:
            patt = "1313"
        elif self.pattern == 4:
            patt = "78-79"
        elif self.pattern == 5:
            patt = "12235224"

        return "VeryBasicSequence{}".format(patt)

    def __repr__(self):
        """"
        To string
        """
        return self.__str__()

    def new(self):
        counter = 0
        # make the very basic sequence
        if self.pattern == 1:
            while counter+1 < self.n:
                self.sequence[counter] = 1
                self.sequence[counter+1] = 2
                counter+=2
        elif self.pattern == 2:
            while counter < self.n:
                if counter % 4 == 0:
                    self.sequence[counter] = 1
                elif counter % 4 == 1:
                    self.sequence[counter] = 2
                elif counter % 4 == 3:
                    self.sequence == 2
                else:
                    self.sequence[counter] = 3
        elif self.pattern == 3:
            if counter % 2 == 0:
                self.sequence[counter] = 1
            else:
                self.sequence[counter] = 3

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

class OneTermSimpleSequence(Sequence):

    def __init__(self, order, lag, determinism=1, n=100000):
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
                    #print("Rejected matrix for having one cycle")
                    goodMatrix = False

        #seed the sequence
        counter = 0
        while counter < self.lag:
            self.sequence[counter] = randint(1, self.order)
            counter+=1
        #get rid of the fixed points
        while counter < self.n:
            where_to = int(self.sequence[counter-self.lag]-1)
            next = self.transition[where_to]
            if next == 0:
                self.sequence[counter] = randint(1, self.order)
            else:
                self.sequence[counter] = next
            counter+=1

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
        return "({},{})-SimpleSequence".format(self.order, self.lag)

    def __repr__(self):
        """"
        To string
        """
        return self.__str__()

    def new(self):
        #make the transition matrix
        self.transition = np.ones(self.order)
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
                    #print("Rejected matrix for having one cycle")
                    goodMatrix = False

        #seed the sequence
        counter = 0
        while counter < self.lag:
            self.sequence[counter] = randint(1, self.order)
            counter+=1
        #get rid of the fixed points
        while counter < self.n:
            where_to = int(self.sequence[counter-self.lag]-1)
            next = self.transition[where_to]
            if next == 0:
                self.sequence[counter] = randint(1, self.order)
            else:
                self.sequence[counter] = next
            counter+=1

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



def main():
    #ts = OneTermSimpleSequence(7,3)
    ts = VeryBasicSequence()
    for i in range(100):
        print("The time series at {} is {}".format(i, ts.get()))
    #ts.plot()

if __name__ == "__main__":
    main()
