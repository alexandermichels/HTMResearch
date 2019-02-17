#!/usr/bin/env python
import copy
import csv
import json
import os
import yaml

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
from math import sqrt

_PARAMS_PATH = "model.yaml"


def createDataOutLink(network, sensorRegionName, regionName):
    """Link sensor region to other region so that it can pass it data."""
    network.link(sensorRegionName, regionName, "UniformLink", "", srcOutput="dataOut", destInput="bottomUpIn")


def createFeedForwardLink(network, regionName1, regionName2):
    """Create a feed-forward link between 2 regions: regionName1 -> regionName2"""
    network.link(regionName1, regionName2, "UniformLink", "", srcOutput="bottomUpOut", destInput="bottomUpIn")


def createResetLink(network, sensorRegionName, regionName):
    """Create a reset link from a sensor region: sensorRegionName -> regionName"""
    network.link(sensorRegionName, regionName, "UniformLink", "", srcOutput="resetOut", destInput="resetIn")


def createSensorToClassifierLinks(network, sensorRegionName, classifierRegionName):
    """Create required links from a sensor region to a classifier region."""
    network.link(sensorRegionName, classifierRegionName, "UniformLink", "", srcOutput="bucketIdxOut", destInput="bucketIdxIn")
    network.link(sensorRegionName, classifierRegionName, "UniformLink", "", srcOutput="actValueOut", destInput="actValueIn")
    network.link(sensorRegionName, classifierRegionName, "UniformLink", "", srcOutput="categoryOut", destInput="categoryIn")

def createEncoder(rdse_resolution):
    """Create the encoder instance for our test and return it."""
    series_rdse = RandomDistributedScalarEncoder(rdse_resolution, name="rdse with resolution {}".format(rdse_resolution))
    encoder = MultiEncoder()
    encoder.addEncoder("series", series_rdse)
    return encoder

def createNetwork(dataSource, rdse_resolution, cellsPerMiniColumn=32):
    """Create the Network instance.

    The network has a sensor region reading data from `dataSource` and passing
    the encoded representation to an SPRegion. The SPRegion output is passed to
    a TMRegion.

    :param dataSource: a RecordStream instance to get data from
    :param cellsPerMiniColumn: int, number of cells per mini-column. Default=32
    :returns: a Network instance ready to run
    """
    try:
        with open(_PARAMS_PATH, "r") as f:
            modelParams = yaml.safe_load(f)["modelParams"]
    except:
        with open(os.path.join("..",_PARAMS_PATH), "r") as f:
            modelParams = yaml.safe_load(f)["modelParams"]

    # Create a network that will hold the regions.
    network = Network()

    # Add a sensor region.
    network.addRegion("sensor", "py.RecordSensor", '{}')

    # Set the encoder and data source of the sensor region.
    sensorRegion = network.regions["sensor"].getSelf()
    #sensorRegion.encoder = createEncoder(modelParams["sensorParams"]["encoders"])
    sensorRegion.encoder = createEncoder(rdse_resolution)
    sensorRegion.dataSource = dataSource

    # Make sure the SP input width matches the sensor region output width.
    modelParams["spParams"]["inputWidth"] = sensorRegion.encoder.getWidth()
    modelParams["tmParams"]["cellsPerColumn"] = cellsPerMiniColumn

    # Add SP and TM regions.
    network.addRegion("spatialPoolerRegion", "py.SPRegion", json.dumps(modelParams["spParams"]))
    network.addRegion("temporalPoolerRegion", "py.TMRegion", json.dumps(modelParams["tmParams"]))

    # Add a classifier region.
    clName = "py.%s" % modelParams["clParams"].pop("regionName")
    network.addRegion("classifier", clName, json.dumps(modelParams["clParams"]))

    # Add all links
    createSensorToClassifierLinks(network, "sensor", "classifier")
    createDataOutLink(network, "sensor", "spatialPoolerRegion")
    createFeedForwardLink(network, "spatialPoolerRegion", "temporalPoolerRegion")
    createFeedForwardLink(network, "temporalPoolerRegion", "classifier")
    # Reset links are optional, since the sensor region does not send resets.
    createResetLink(network, "sensor", "spatialPoolerRegion")
    createResetLink(network, "sensor", "temporalPoolerRegion")

    # Make sure all objects are initialized.
    network.initialize()

    return network

def getPredictionResults(network, clRegionName):
    """Helper function to extract results for all prediction steps."""


    classifierRegion = network.regions[clRegionName]
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


def runNetwork(network,learning = True):
    """Run the network and write output to writer.

    :param network: a Network instance to run
    :param writer: a csv.writer instance to write output to
    """
    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
    _OUTPUT_PATH = "../outputs/HTMOutput-{}-{}.csv".format(DATE, time_series_model)


    sensorRegion = network.regions["sensor"]
    spatialPoolerRegion = network.regions["spatialPoolerRegion"]
    temporalPoolerRegion = network.regions["temporalPoolerRegion"]

    # Set predicted field
    network.regions["sensor"].setParameter("predictedField", "series")

    if learning:
        # Enable learning for all regions.
        network.regions["spatialPoolerRegion"].setParameter("learningMode", 1)
        network.regions["temporalPoolerRegion"].setParameter("learningMode", 1)
        network.regions["classifier"].setParameter("learningMode", 1)
    else:
        # Enable learning for all regions.
        network.regions["spatialPoolerRegion"].setParameter("learningMode", 0)
        network.regions["temporalPoolerRegion"].setParameter("learningMode", 0)
        network.regions["classifier"].setParameter("learningMode", 0)


    # Enable inference for all regions.
    network.regions["spatialPoolerRegion"].setParameter("inferenceMode", 1)
    network.regions["temporalPoolerRegion"].setParameter("inferenceMode", 1)
    network.regions["classifier"].setParameter("inferenceMode", 1)

    _model = network.regions["sensor"].getSelf().dataSource

    with open(_OUTPUT_PATH, "w") as outputFile:
        writer = csv.writer(outputFile)
        print("Writing output to {}".format(_OUTPUT_PATH))
        writer.writerow(["Time Step", "Series", "One Step Prediction", "One Step Prediction Confidence", "Five Step Prediction", "Five Step Prediction Confidence"])
        results = []
        for i in range(len(_model)):
            # Run the network for a single iteration
            network.run(1)

            series = sensorRegion.getOutputData("sourceOut")[0]
            predictionResults = getPredictionResults(network, "classifier")
            oneStep = predictionResults[1]["predictedValue"]
            oneStepConfidence = predictionResults[1]["predictionConfidence"]
            fiveStep = predictionResults[5]["predictedValue"]
            fiveStepConfidence = predictionResults[5]["predictionConfidence"]

            result = [_model.getBookmark(), series, oneStep, oneStepConfidence*100, fiveStep, fiveStepConfidence*100]
            #print "{:6}: 1-step: {:16} ({:4.4}%)\t 5-step: {:16} ({:4.4}%)".format(*result)
            results.append(result)
            writer.writerow(result)
        return results

def runNetworkWithMode(network, mode, eval_method="val", error_method = "MSE"):
    '''
    Modes:
    * "strain" - Learning on spatial pool, on training set
    * "train" - Learning, on training set
    * "test" - No learning, on test set
    * "eval" - Learning, on eval set
    '''
    _model = network.regions["sensor"].getSelf().dataSource
    sensorRegion = network.regions["sensor"]
    # Set predicted field
    network.regions["sensor"].setParameter("predictedField", "series")

    if mode == "strain":
        _model.set_to_train_theta()
        # Enable learning for all regions.
        network.regions["spatialPoolerRegion"].setParameter("learningMode", 1)
        network.regions["temporalPoolerRegion"].setParameter("learningMode", 0)
        network.regions["classifier"].setParameter("learningMode", 0)
        # Enable inference for all regions.
        network.regions["spatialPoolerRegion"].setParameter("inferenceMode", 1)
        network.regions["temporalPoolerRegion"].setParameter("inferenceMode", 1)
        network.regions["classifier"].setParameter("inferenceMode", 1)
        result = 0
        last_prediction = None
        count = 0
        while _model.in_train_set() and count < 1000:
            network.run(1)
            series = sensorRegion.getOutputData("sourceOut")[0]
            predictionResults = getPredictionResults(network, "classifier")
            if last_prediction != None:
                if error_method == "MSE":

                    result+=sqrt((series-last_prediction)**2)
                elif error_method == "Binary":
                    if series==last_prediction:
                        result+= 0
                    else:
                        result+= 1
            last_prediction=predictionResults[1]["predictedValue"]
            count+=1
        return result
    elif mode == "train":
        _model.set_to_train_theta()
        # Enable learning for all regions.
        network.regions["spatialPoolerRegion"].setParameter("learningMode", 1)
        network.regions["temporalPoolerRegion"].setParameter("learningMode", 1)
        network.regions["classifier"].setParameter("learningMode", 1)
        # Enable inference for all regions.
        network.regions["spatialPoolerRegion"].setParameter("inferenceMode", 1)
        network.regions["temporalPoolerRegion"].setParameter("inferenceMode", 1)
        network.regions["classifier"].setParameter("inferenceMode", 1)
        result = 0
        last_prediction = None
        count = 0
        while _model.in_train_set() and count < 1000:
            network.run(1)
            series = sensorRegion.getOutputData("sourceOut")[0]
            predictionResults = getPredictionResults(network, "classifier")
            if last_prediction != None:
                if error_method == "MSE":
                    result+=sqrt((series-last_prediction)**2)
                elif error_method == "Binary":
                    if series==last_prediction:
                        result+= 0
                    else:
                        result+=1
            last_prediction=predictionResults[1]["predictedValue"]
            count+=1
        return result
    elif mode == "test":
        _model.set_to_test_theta()
        # Disable learning for all regions.
        network.regions["spatialPoolerRegion"].setParameter("learningMode", 0)
        network.regions["temporalPoolerRegion"].setParameter("learningMode", 0)
        network.regions["classifier"].setParameter("learningMode", 0)
        # Enable inference for all regions.
        network.regions["spatialPoolerRegion"].setParameter("inferenceMode", 1)
        network.regions["temporalPoolerRegion"].setParameter("inferenceMode", 1)
        network.regions["classifier"].setParameter("inferenceMode", 1)
        result = 0
        last_prediction = None
        while _model.in_test_set():
            network.run(1)
            series = sensorRegion.getOutputData("sourceOut")[0]
            predictionResults = getPredictionResults(network, "classifier")
            if last_prediction != None:
                if error_method == "MSE":
                    result+=sqrt((series-last_prediction)**2)
                elif error_method == "Binary":
                    if series==last_prediction:
                        result+= 0
                    else:
                        result+=1
            last_prediction=predictionResults[1]["predictedValue"]
        return result
    elif mode == "eval":
        _model.set_to_eval_theta()
        # Disable learning for all regions.
        network.regions["spatialPoolerRegion"].setParameter("learningMode", 0)
        network.regions["temporalPoolerRegion"].setParameter("learningMode", 0)
        network.regions["classifier"].setParameter("learningMode", 0)
        # Enable inference for all regions.
        network.regions["spatialPoolerRegion"].setParameter("inferenceMode", 1)
        network.regions["temporalPoolerRegion"].setParameter("inferenceMode", 1)
        network.regions["classifier"].setParameter("inferenceMode", 1)
        if eval_method == "val":
            result = 0
        elif eval_method == "expressive":
            result = []
        last_prediction = None
        while _model.in_eval_set():
            network.run(1)
            series = sensorRegion.getOutputData("sourceOut")[0]
            predictionResults = getPredictionResults(network, "classifier")
            if eval_method == "val":
                if last_prediction != None and error_method == "MSE":
                    result+=sqrt((series-last_prediction)**2)
                elif last_prediction != None and error_method == "Binary":
                    if series==last_prediction:
                        result+= 0
                    else:
                        result+=1
            elif eval_method == "expressive":
                oneStep = predictionResults[1]["predictedValue"]
                oneStepConfidence = predictionResults[1]["predictionConfidence"]
                fiveStep = predictionResults[5]["predictedValue"]
                fiveStepConfidence = predictionResults[5]["predictionConfidence"]

                result.append([_model.getBookmark(), series, oneStep, oneStepConfidence*100, fiveStep, fiveStepConfidence*100])
                #print "{:6}: 1-step: {:16} ({:4.4}%)\t 5-step: {:16} ({:4.4}%)".format(*result)
            last_prediction=predictionResults[1]["predictedValue"]
        return result
    else:
        print("No valid mode selected")

def HTM(time_series_model, rdse_resolution=1, cellsPerMiniColumn=None, verbosity=1):
    if cellsPerMiniColumn == None:
        network = createNetwork(TimeSeriesStream(time_series_model), rdse_resolution)
    else:
        network = createNetwork(TimeSeriesStream(time_series_model), rdse_resolution, cellsPerMiniColumn)
    network.initialize()

    spRegion = network.getRegionsByType(SPRegion)[0]
    sp = spRegion.getSelf().getAlgorithmInstance()
    if verbosity > 1:
        print("spatial pooler region inputs: {0}".format(spRegion.getInputNames()))
        print("spatial pooler region outputs: {0}".format(spRegion.getOutputNames()))
        print("# spatial pooler columns: {0}\n".format(sp.getNumColumns()))

    tmRegion = network.getRegionsByType(TMRegion)[0]
    tm = tmRegion.getSelf().getAlgorithmInstance()
    if verbosity > 1:
        print("temporal memory region inputs: {0}".format(tmRegion.getInputNames()))
        print("temporal memory region outputs: {0}".format(tmRegion.getOutputNames()))
        print("# temporal memory columns: {0}".format(tm.numberOfCols))

    return network

def train(network, eval_method="val", error_method="MSE"):
    last_error = float("inf") # set to infinity error so you keep training the first time
    curr_error = float("inf")
    counter = 0
    for i in range(25):
        runNetworkWithMode(network, "strain", "val", error_method)
    while (curr_error <= last_error and counter <20):
        last_error=curr_error
        curr_error = 0
        runNetworkWithMode(network, "train", "val", error_method)
        curr_error = runNetworkWithMode(network, "test", "val", error_method)
        counter+=1
    return runNetworkWithMode(network, "eval", eval_method, error_method)

if __name__ == "__main__":

    #carried over from another file, so I have the baseline
    #parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('-d', type=str, required=True, dest='dataset', help='Dataset to test on. ')
    #parser.add_argument('-o', type=str, required=True, dest='outdir', help='Path to the output directory.')
    #parser.add_argument('-m', type=str, required=True, dest='method', help='Method to use: stream, relklinker, klinker, predpath, sm')
    #args = parser.parse_args()
    time_series_model = ARMATimeSeries(2,0)
    network = HTM(time_series_model, cellsPerMiniColumn=2)
    print(train(network))
