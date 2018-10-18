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

def createEncoder():
    """Create the encoder instance for our test and return it."""
    series_rdse = RandomDistributedScalarEncoder(0.01, name="rdse with resolution 0.01")
    encoder = MultiEncoder()
    encoder.addEncoder("series", series_rdse)
    return encoder

def createNetwork(dataSource, cellsPerMiniColumn=32):
    """Create the Network instance.

    The network has a sensor region reading data from `dataSource` and passing
    the encoded representation to an SPRegion. The SPRegion output is passed to
    a TMRegion.

    :param dataSource: a RecordStream instance to get data from
    :param cellsPerMiniColumn: int, number of cells per mini-column. Default=32
    :returns: a Network instance ready to run
    """
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)["modelParams"]

    # Create a network that will hold the regions.
    network = Network()

    # Add a sensor region.
    network.addRegion("sensor", "py.RecordSensor", '{}')

    # Set the encoder and data source of the sensor region.
    sensorRegion = network.regions["sensor"].getSelf()
    #sensorRegion.encoder = createEncoder(modelParams["sensorParams"]["encoders"])
    sensorRegion.encoder = createEncoder()
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


def runNetwork(network, writer):
    """Run the network and write output to writer.

    :param network: a Network instance to run
    :param writer: a csv.writer instance to write output to
    """
    sensorRegion = network.regions["sensor"]
    spatialPoolerRegion = network.regions["spatialPoolerRegion"]
    temporalPoolerRegion = network.regions["temporalPoolerRegion"]

    # Set predicted field
    network.regions["sensor"].setParameter("predictedField", "series")

    # Enable learning for all regions.
    network.regions["spatialPoolerRegion"].setParameter("learningMode", 1)
    network.regions["temporalPoolerRegion"].setParameter("learningMode", 1)
    network.regions["classifier"].setParameter("learningMode", 1)

    # Enable inference for all regions.
    network.regions["spatialPoolerRegion"].setParameter("inferenceMode", 1)
    network.regions["temporalPoolerRegion"].setParameter("inferenceMode", 1)
    network.regions["classifier"].setParameter("inferenceMode", 1)

    _model = network.regions["sensor"].getSelf().dataSource

    writer.writerow(["Time Step", "Series", "One Step Prediction", "One Step Prediction Confidence", "Five Step Prediction", "Five Step Prediction Confidence"])
    results = []
    for i in tqdm(range(len(_model))):
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

def HTM(time_series_model, cellsPerMiniColumn=None):
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)["modelParams"]

    if cellsPerMiniColumn == None:
        network = createNetwork(TimeSeriesStream(time_series_model))
    else:
        network = createNetwork(TimeSeriesStream(time_series_model), cellsPerMiniColumn)
    network.initialize()

    spRegion = network.getRegionsByType(SPRegion)[0]
    sp = spRegion.getSelf().getAlgorithmInstance()
    print "spatial pooler region inputs: {0}".format(spRegion.getInputNames())
    print "spatial pooler region outputs: {0}".format(spRegion.getOutputNames())
    print "# spatial pooler columns: {0}".format(sp.getNumColumns())
    print

    tmRegion = network.getRegionsByType(TMRegion)[0]
    tm = tmRegion.getSelf().getAlgorithmInstance()
    print "temporal memory region inputs: {0}".format(tmRegion.getInputNames())
    print "temporal memory region outputs: {0}".format(tmRegion.getOutputNames())
    print "# temporal memory columns: {0}".format(tm.numberOfCols)
    print

    DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
    _OUTPUT_PATH = "../outputs/HTMOutput-{}-{}-{}.csv".format(DATE, cellsPerMiniColumn, time_series_model)

    with open(_OUTPUT_PATH, "w") as outputFile:
        writer = csv.writer(outputFile)
        print "Writing output to %s" % _OUTPUT_PATH
        result = runNetwork(network, writer)
    return result


if __name__ == "__main__":

    #carried over from another file, so I have the baseline
    #parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('-d', type=str, required=True, dest='dataset', help='Dataset to test on. ')
    #parser.add_argument('-o', type=str, required=True, dest='outdir', help='Path to the output directory.')
    #parser.add_argument('-m', type=str, required=True, dest='method', help='Method to use: stream, relklinker, klinker, predpath, sm')
    #args = parser.parse_args()
    time_series_model = ARMATimeSeries(2,0)
    HTM(time_series_model, cellsPerMiniColumn=32)
