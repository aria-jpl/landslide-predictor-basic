#!/bin/bash

set -e

BASE_PATH=/home/ops

echo "run plot"

date

cd ${BASE_PATH}/plot

lonlatFilePath=${BASE_PATH}/timeseries/ps.lonlat.npy
predictionFilePath=${BASE_PATH}/predictor/output/ps.displacement.pred
outputFilePath=./ps.prediction.png
sh ./run.sh $lonlatFilePath $predictionFilePath $outputFilePath

date
