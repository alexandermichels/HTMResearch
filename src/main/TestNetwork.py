#!/usr/bin/env python
import copy
import csv
import json
import os
import yaml
import numpy

from pkg_resources import resource_filename

from nupic.data.file_record_stream import FileRecordStream
from nupic.engine import Network
from nupic.encoders import MultiEncoder, ScalarEncoder, DateEncoder
from nupic.regions.sp_region import SPRegion
from nupic.regions.tm_region import TMRegion

from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
from models.ARMAModels import ARMATimeSeries
from TimeSeriesStream import TimeSeriesStream
from time import localtime, strftime
import argparse
from tqdm import tqdm

_PARAMS_PATH = "model.yaml"

class HTM():

    def __init__(self, rdse_resolution=.5):
        self.createEncoder(rdse_resolution)

    def encode(self, inputData):
        return self.encoder.encode(inputData)

    def createEncoder(self, rdse_resolution):
        """Create the encoder instance for our test and return it."""
        self.encoder = RandomDistributedScalarEncoder(rdse_resolution, name="rdse with resolution {}".format(rdse_resolution))

    def overlap(self, inputA, inputB):
        return numpy.sum(numpy.multiply(self.encoder.encode(inputA), self.encoder.encode(inputB)))




if __name__ == "__main__":
    l = [1,2,2,3,1,2,2,3,1,2,2,3,1,2,2,3,1,2,2,3,1,2,2,3,1,2,2,3,1,2,2,3,1,2,2,3,1,2,2,3,1,2,2,3,1,2,2,3,1,2,2,3,1,2,2,3]
    for i in range(10):
        h = HTM(rdse_resolution=(i+1)*.01)
        str = "{}:   ".format((i+1)*.01)
        for j in range(len(l)-1):
            str+="{}, ".format(h.overlap(l[j],l[j+1]))
        print("{}\n".format(str[:-2]))
