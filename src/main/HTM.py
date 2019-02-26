#!/usr/bin/env python
""" Standard Packages"""
import argparse, csv, copy, json, numpy, os, sys, yaml
from pkg_resources import resource_filename
import logging as log
from os.path import join
import numpy as np
numpy.set_printoptions(threshold=sys.maxsize)
from time import localtime, strftime
from tqdm import tqdm
from math import sqrt

""" Nupic Imports"""
from nupic.data.file_record_stream import FileRecordStream
from nupic.engine import Network
from nupic.encoders import MultiEncoder, ScalarEncoder, DateEncoder
from nupic.regions.sp_region import SPRegion
from nupic.regions.tm_region import TMRegion
from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder

""" My Stuff """
from models.SimpleSequence import VeryBasicSequence
from models.ARMAModels import ARMATimeSeries
from TimeSeriesStream import TimeSeriesStream


_PARAMS_PATH = "model.yaml"

def fcompare(arg1, arg2, TOL=.001):
    """
    A helper function to compare epsilon comparisons for floating point numbers

    Returns:
    * 1 if arg1 > arg2
    * 0 if arg1 == arg2
    * -1 if arg1 < arg2
    """
    diff = arg1-arg2 # calculate the difference between arg1 and arg2
    if abs(diff) <= TOL: # if the absolute value of the difference is within epsilon, they are the same
        return 0
    elif diff > TOL: # arg1 is bigger
        return 1
    return -1 # arg2 is bigger

class RDSEEncoder():

    def __init__(self, resolution=.5):
        """Create the encoder instance for our test and return it."""
        self.resolution = resolution
        self.series_encoder = RandomDistributedScalarEncoder(self.resolution, name="RDSE-(res={})".format(self.resolution))
        self.encoder = MultiEncoder()
        self.encoder.addEncoder("series", self.series_encoder)
        self.last_m_encode = np.zeros(1)

    def get_encoder(self):
        return self.encoder

    def get_resolution(self):
        return self.resolution

    def m_encode(self, inputData):
        self.last_m_encode = self.encoder.encode(inputData)
        return self.last_m_encode

    def m_overlap(self, inputData):
        temp = self.last_m_encode
        self.last_m_encode = self.encoder.encode(inputData)
        return numpy.sum(numpy.multiply(self.last_m_encode, temp))

    def r_encode(self, inputData):
        return self.series_encoder.encode(inputData)

    def r_overlap(self, inputA, inputB):
        return numpy.sum(numpy.multiply(self.series_encoder.encode(inputA), self.series_encoder.encode(inputB)))


# object for Spatial Pooler with Params

# object for Temporal Pooler with Params

class HTM():

    global param_dict
    param_dict = { "tmParams" : { "cellsPerColumn": 4 } }

    def __init__(self, dataSource, rdse_resolution, params=None, verbosity=3):
        """Create the Network instance.

        The network has a sensor region reading data from `dataSource` and passing
        the encoded representation to an SPRegion. The SPRegion output is passed to
        a TMRegion.

        :param dataSource: a RecordStream instance to get data from
        :param rdse_resolution: float, resolution of Random Distributed Scalar Encoder
        :param cellsPerMiniColumn: int, number of cells per mini-column. Default=32
        """
        if verbosity > 0:
            DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
            self.log_file = join('../logs/', 'HTM-{}-({}RDSEres)-datasource-{}.log'.format(DATE,rdse_resolution,str(dataSource)))
            log.basicConfig(format = '[%(asctime)s] %(message)s', datefmt = '%m/%d/%Y %H:%M:%S %p', filename = self.log_file, level=log.DEBUG)
            log.getLogger().addHandler(log.StreamHandler())
            self.setVerbosity(verbosity)
        else:
            self.log_file = None

        self.modelParams = {}
        log.debug("...loading params from {}...".format(_PARAMS_PATH))
        try:
            with open(_PARAMS_PATH, "r") as f:
                self.modelParams = yaml.safe_load(f)["modelParams"]
        except:
            with open(os.path.join("..",_PARAMS_PATH), "r") as f:
                self.modelParams = yaml.safe_load(f)["modelParams"]
        # Create a network that will hold the regions.
        self.network = Network()
        # Add a sensor region.
        self.network.addRegion("sensor", "py.RecordSensor", '{}')
        # Set the encoder and data source of the sensor region.
        self.sensorRegion = self.network.regions["sensor"].getSelf()
        #sensorRegion.encoder = createEncoder(modelParams["sensorParams"]["encoders"])
        self.encoder = RDSEEncoder(rdse_resolution)
        self.sensorRegion.encoder = self.encoder.get_encoder()
        self.sensorRegion.dataSource = TimeSeriesStream(dataSource)
        self.network.regions["sensor"].setParameter("predictedField", "series")

        # Adjust params
        # Make sure the SP input width matches the sensor region output width.
        self.modelParams["spParams"]["inputWidth"] = self.sensorRegion.encoder.getWidth()
        if not params == None:
            for key, value in params.iteritems():
                if key == "clParams" or key == "spParams" or key == "tmParams":
                    for vkey, vvalue in value.iteritems():
                        #print(key, vkey, vvalue)
                        self.modelParams[key][vkey] = vvalue
        log.debug("xxx HTM Params: xxx\n{}\n".format(json.dumps(self.modelParams, sort_keys=True, indent=4)))
        # Add SP and TM regions.
        self.network.addRegion("spatialPoolerRegion", "py.SPRegion", json.dumps(self.modelParams["spParams"]))
        self.network.addRegion("temporalPoolerRegion", "py.TMRegion", json.dumps(self.modelParams["tmParams"]))
        # Add a classifier region.
        clName = "py.%s" % self.modelParams["clParams"].pop("regionName")
        self.network.addRegion("classifier", clName, json.dumps(self.modelParams["clParams"]))
        # link regions
        self.linkSensorToClassifier()
        self.linkSensorToSpatialPooler()
        self.linkSpatialPoolerToTemporalPooler()
        self.linkTemporalPoolerToClassifier()
        self.linkResets()
        # possibly do reset links here (says they are optional
        self.network.initialize()
        self.turnInferenceOn()
        self.turnLearningOn()

    def __str__(self):
        spRegion = self.network.getRegionsByType(SPRegion)[0]
        sp = spRegion.getSelf().getAlgorithmInstance()
        _str = "spatial pooler region inputs: {0}\n".format(spRegion.getInputNames())
        _str+="spatial pooler region outputs: {0}\n".format(spRegion.getOutputNames())
        _str+="# spatial pooler columns: {0}\n\n".format(sp.getNumColumns())

        tmRegion = self.network.getRegionsByType(TMRegion)[0]
        tm = tmRegion.getSelf().getAlgorithmInstance()
        _str+="temporal memory region inputs: {0}\n".format(tmRegion.getInputNames())
        _str+="temporal memory region outputs: {0}\n".format(tmRegion.getOutputNames())
        _str+="# temporal memory columns: {0}\n".format(tm.numberOfCols)
        return _str

    def getClassifierResults(self):
        """Helper function to extract results for all prediction steps."""
        classifierRegion = self.network.regions["classifier"]
        actualValues = classifierRegion.getOutputData("actualValues")
        probabilities = classifierRegion.getOutputData("probabilities")
        steps = classifierRegion.getSelf().stepsList
        N = classifierRegion.getSelf().maxCategoryCount
        results = {step: {} for step in steps}
        for i in range(len(steps)):
            # stepProbabilities are probabilities for this prediction step only.
            stepProbabilities = probabilities[i * N:(i + 1) * N - 1]
            mostLikelyCategoryIdx = stepProbabilities.argmax()
            predictedValue = actualValues[mostLikelyCategoryIdx]
            predictionConfidence = stepProbabilities[mostLikelyCategoryIdx]
            results[steps[i]]["predictedValue"] = predictedValue
            results[steps[i]]["predictionConfidence"] = predictionConfidence
        return results

    def getCurrSeries(self):
        return self.network.regions["sensor"].getOutputData("sourceOut")[0]

    def getTimeSeriesStream(self):
        return self.network.regions["sensor"].getSelf().dataSource

    def linkResets(self):
        """createResetLink(network, "sensor", "spatialPoolerRegion")
        createResetLink(network, "sensor", "temporalPoolerRegion")"""
        self.network.link("sensor", "spatialPoolerRegion", "UniformLink", "", srcOutput="resetOut", destInput="resetIn")
        self.network.link("sensor", "temporalPoolerRegion", "UniformLink", "", srcOutput="resetOut", destInput="resetIn")

    def linkSensorToClassifier(self):
        """Create required links from a sensor region to a classifier region."""
        self.network.link("sensor", "classifier", "UniformLink", "", srcOutput="bucketIdxOut", destInput="bucketIdxIn")
        self.network.link("sensor", "classifier", "UniformLink", "", srcOutput="actValueOut", destInput="actValueIn")
        self.network.link("sensor", "classifier", "UniformLink", "", srcOutput="categoryOut", destInput="categoryIn")

    def linkSensorToSpatialPooler(self):
        self.network.link("sensor", "spatialPoolerRegion", "UniformLink", "", srcOutput="dataOut", destInput="bottomUpIn")

    def linkSpatialPoolerToTemporalPooler(self):
        """Create a feed-forward link between 2 regions: spatialPoolerRegion -> temporalPoolerRegion"""
        self.network.link("spatialPoolerRegion", "temporalPoolerRegion", "UniformLink", "", srcOutput="bottomUpOut", destInput="bottomUpIn")

    def linkTemporalPoolerToClassifier(self):
        """Create a feed-forward link between 2 regions: temporalPoolerRegion -> classifier"""
        self.network.link("temporalPoolerRegion", "classifier", "UniformLink", "", srcOutput="bottomUpOut", destInput="bottomUpIn")

    def setVerbosity(self, level):
        """
        Sets the level of print statements/logging (verbosity)
        * 3 == DEBUG
        * 2 == VERBOSE
        * 1 == WARNING
        """
        if self.log_file == None: # if there's no log file, make one
            DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
            self.log_file = join('../logs/', 'HTM-{}-({}CPMC-{}RDSEres)-datasource-{}.log'.format(DATE,self.modelParams["tmParams"]["cellsPerColumn"],self.encoder.get_resolution(),str(self.sensorRegion.dataSource)))
            log.basicConfig(format = '[%(asctime)s] %(message)s', datefmt = '%m/%d/%Y %H:%M:%S %p', filename = self.log_file, level=log.DEBUG)
            log.getLogger().addHandler(log.StreamHandler())

        if level >= 3:
            log.getLogger().setLevel(log.DEBUG)
        elif level >= 2:
            log.getLogger().setLevel(log.VERBOSE)
        elif level >= 1:
            log.getLogger().setLevel(log.WARNING)

    def runNetwork(self, learning = True):
        DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
        _OUTPUT_PATH = "../outputs/HTMOutput-{}-{}.csv".format(DATE, self.network.regions["sensor"].getSelf().dataSource)
        self.sensorRegion.dataSource.rewind()

        # Set predicted field
        self.network.regions["sensor"].setParameter("predictedField", "series")

        if learning:
            # Enable learning for all regions.
            self.turnLearningOn()
        else:
            # Enable learning for all regions.
            self.turnLearningOff()
        self.turnInferenceOn()

        _model = self.network.regions["sensor"].getSelf().dataSource

        with open(_OUTPUT_PATH, "w") as outputFile:
            writer = csv.writer(outputFile)
            log.info("Writing output to {}".format(_OUTPUT_PATH))
            steps = self.network.regions["classifier"].getSelf().stepsList
            header_row = ["Time Step", "Series"]
            for step in steps:
                header_row.append("{} Step Pred".format(step))
                header_row.append("{} Step Pred Conf".format(step))
            writer.writerow(header_row)
            results = []
            for i in range(len(_model)):
                # Run the network for a single iteration
                self.network.run(1)

                series = self.network.regions["sensor"].getOutputData("sourceOut")[0]
                predictionResults = self.getClassifierResults()
                result = [_model.getBookmark(), series ]
                for key, value in predictionResults.iteritems():
                    result.append(value["predictedValue"])
                    result.append(value["predictionConfidence"]*100)
                #print "{:6}: 1-step: {:16} ({:4.4}%)\t 5-step: {:16} ({:4.4}%)".format(*result)
                results.append(result)
                writer.writerow(result)
                outputFile.flush()
            return results


    def runWithMode(self, mode, eval_method="val", error_method="rmse", five_weight=1.0):
        '''
        Modes:
        * "strain" - Learning on spatial pool, on training set
        * "train" - Learning, on training set
        * "test" - No learning, on test set
        * "eval" - Learning, on eval set
        '''
        mode = mode.lower()
        eval_method = eval_method.lower()
        error_method = error_method.lower()
        log.debug("enetered `runWithMode` with with:\n  mode: {}\n  eval_method: {}\n  error_method: {}".format(mode, eval_method, error_method))

        _model = self.getTimeSeriesStream()

        if mode == "strain":
            self.turnLearningOff("t")
            self.turnLearningOn("cs")
        else:
            self.turnLearningOn()
        self.turnInferenceOn()

        if eval_method == "val":
            log.debug("val evaluation method selected in `runWithMode`")
            result = 0
        elif eval_method == "expr":
            log.debug("expr (expressive) evaluation method selected in `runWithMode`")
            result = []

        last_prediction = None
        five_pred = [None]*5 # list of 5 Nones
        if mode == "strain" or mode == "train":
            _model.set_to_train_theta()
            while _model.in_train_set():
                temp = self.run_with_mode_one_iter("val", error_method, result, five_weight, last_prediction, five_pred)
                result = temp[0]
                last_prediction = temp[1]
                five_pred = temp[2]
        elif mode == "test":
            _model.set_to_test_theta()
            while _model.in_test_set():
                temp = self.run_with_mode_one_iter("val", error_method, result, five_weight, last_prediction, five_pred)
                result = temp[0]
                last_prediction = temp[1]
                five_pred = temp[2]
        elif mode == "eval":
            _model.set_to_eval_theta()
            while _model.in_eval_set():
                temp = self.run_with_mode_one_iter(eval_method, error_method, result, 0, last_prediction, five_pred)
                result = temp[0]
                last_prediction = temp[1]
                five_pred = temp[2]

            # normalize result over length of evaluation set
            if eval_method=="val":
                result/=(2*self.sensorRegion.dataSource.len_eval_set()-2)

        return result

    def run_with_mode_one_iter(self, eval_method, error_method, result, five_weight, last_prediction, five_pred):
        self.network.run(1)
        series = self.getCurrSeries()
        predictions = self.getClassifierResults()

        if eval_method == "val":
            if last_prediction == None:
                pass
            elif error_method == "rmse":
                result+=sqrt((series-last_prediction)**2)
                if not five_pred[0] == None:
                    result+=(five_weight*sqrt((series-five_pred[0])**2))
            elif error_method == "binary":
                if series==last_prediction:
                    result+=0
                else:
                    result+=1
                if not five_pred[0] == None:
                    if series==five_pred[0]:
                        result+=0
                    else:
                        result+=five_weight
        elif error_method == "expr":
            temp = [_model.getBookmark(), series ]
            steps = self.network.regions["classifier"].getSelf().stepsList
            for step in steps:
                temp.append(predictions[i]["predictedValue"])
                temp.append(predictions[i]["predictionConfidence"]*100)
            result.append(temp)

        # update predictions
        classRes = self.getClassifierResults()
        last_prediction = classRes[1]["predictedValue"]
        for i in range(4):
            five_pred[i] = five_pred[i+1] # shift down
        five_pred[4] = classRes[5]["predictedValue"]

        return (result, last_prediction, five_pred)

    def train(self, eval_method="val", error_method="rmse", sibt=3, iter_per_cycle=2, max_cycles=50, five_weight=1.0, log=False):
        """
        Trains the HTM on `dataSource`

        :param  eval_method - the kind of evaluation you'd like (a single value or "expr"
        :param  error_method - the metric for calculating error ("rmse" root mean squared error or "binary")
        :param  sibt - spatial (pooler) iterations before temporal (pooler)
        """
        if log:
            for i in range(sibt):
                log.debug("\nxxxxx Iteration {}/{} of the Spatial Pooler Training xxxxx".format(i+1, sibt))
                # train on spatial pooler
                log.debug("Error for spatial training iteration {} was {} with {} error method".format(i,self.runWithMode("strain", "val", error_method, five_weight), error_method))
            log.info("\nExited spatial pooler only training loop")
        last_error = 0 # set to infinity error so you keep training the first time
        curr_error = -1
        counter = 0
        if log:
            log.info("Entering full training loop")
        while (fcompare(curr_error, last_error) == -1 and counter < max_cycles):
            if log:
                log.debug("\n++++++++++ Cycle {} of the full training loop +++++++++\n".format(counter))
            last_error=curr_error
            curr_error = 0
            for i in range(int(iter_per_cycle)):
                if log:
                    log.debug("\n----- Iteration {}/{} of Cycle {} -----\n".format(i+1, iter_per_cycle, counter))
                    log.debug("Error for full training cycle {}, iteration {} was {} with {} error method".format(counter,i,self.runWithMode("train", "val", error_method, five_weight), error_method))
                curr_error+=self.runWithMode("test", "val", error_method, five_weight)
            if log:
                log.debug("Cycle {} - last: {}    curr: {}".format(counter, last_error, curr_error))
            counter+=1
            if last_error == -1:
                last_error = float("inf")
        self.sensorRegion.dataSource.rewind()
        return self.runWithMode("eval", eval_method, error_method, five_weight)

    def turnInferenceOn(self):
        log.debug("Inference enabled for all regions")
        self.network.regions["spatialPoolerRegion"].setParameter("inferenceMode", 1)
        self.network.regions["temporalPoolerRegion"].setParameter("inferenceMode", 1)
        self.network.regions["classifier"].setParameter("inferenceMode", 1)

    def turnLearningOn(self, turnOn="cst"):
        """
        Turns learning on for certain segments

        :param turnOn - a string of characters representing the segments you'd like to turn on
        * c ---> classifier
        * s ---> spatial pooler
        * t ---> temporal pooler
        """
        for i in range(len(turnOn)):
            target=turnOn[0].lower()
            turnOn=turnOn[1:]
            if target == "c":
                log.debug("Learning enabled for classifier")
                self.network.regions["classifier"].setParameter("learningMode", 1)
            elif target == "s":
                log.debug("Learning enabled for spatial pooler region")
                self.network.regions["spatialPoolerRegion"].setParameter("learningMode", 1)
            elif target == "t":
                log.debug("Learning enabled for temporal pooler region")
                self.network.regions["temporalPoolerRegion"].setParameter("learningMode", 1)

    def turnLearningOff(self, turnOff="cst"):
        """
        Turns learning off for certain segments

        :param turnOff - a string of characters representing the segments you'd like to turn off
        * c ---> classifier
        * s ---> spatial pooler
        * t ---> temporal pooler
        """
        for i in range(len(turnOff)):
            target=turnOff[0].lower()
            turnOff=turnOff[1:]
            if target == "c":
                log.debug("Learning disabled for classifier")
                self.network.regions["classifier"].setParameter("learningMode", 0)
            elif target == "s":
                log.debug("Learning disabled for spatial pooler region")
                self.network.regions["spatialPoolerRegion"].setParameter("learningMode", 0)
            elif target == "t":
                log.debug("Learning disabled for temporal pooler region")
                self.network.regions["temporalPoolerRegion"].setParameter("learningMode", 0)


if __name__ == "__main__":
    time_series_model = VeryBasicSequence(pattern=1, n=1000)
    network = HTM(time_series_model, .6)
    print(network)
    #print(network.train(error_method="binary"))
    network.train("val", "binary")
    network.runNetwork()
    '''print(network.network.regions["spatialPoolerRegion"].__dict__)
    print(network.network.regions["spatialPoolerRegion"].getInputNames())
    print(network.network.regions["spatialPoolerRegion"].getInputData("bottomUpIn"))
    print(network.network.regions["spatialPoolerRegion"].getOutputNames())'''
    """l = VeryBasicSequence()
    for i in range(10):
        h = RDSEEncoder(resolution=(i+1)*.01)
        str = "{}:   ".format((i+1)*.01)
        for j in range(len(l)//100):
            str+="{}, ".format(h.r_overlap(l[j],l[j+1]))
        print("{}\n".format(str[:-2]))
    t = TimeSeriesStream(l)
    for i in range(10):
        h = RDSEEncoder(resolution=(i+1)*.01)
        str = "{}:   ".format((i+1)*.01)
        for j in range(len(l)//100):
            str+="{}, ".format(h.m_overlap(t.getNextRecordDict()))
        print("{}\n".format(str[:-2]))"""
