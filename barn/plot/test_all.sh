#!/bin/bash

set -e

BASE_PATH=/home/ops

echo "run plot_prediction_all.sh"

date

cd ${BASE_PATH}/plot

fileNamePrefix="ps"
lonlatDirPath=${BASE_PATH}/timeseries
predictionDirPath=${BASE_PATH}/predictor/output
outputFilePath=./${fileNamePrefix}.all.prediction.png
sh ./plot_prediction_all.sh $fileNamePrefix $lonlatDirPath $predictionDirPath $outputFilePath

date
