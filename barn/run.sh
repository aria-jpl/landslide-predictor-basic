#!/bin/bash

set -e

# prepare timeseries
(
date

cd /home/ops/timeseries

inputDirPath=/home/xing/hysds/stamps/data/apt-stamps-test/INSAR_20171004

sh ./run.sh $inputDirPath

# copy output to input dir of predictor
pickleFileName=displacement.pickle
mv ./$pickleFileName /home/ops/predictor/input

date
)

# run predictor
(
date

cd /home/ops/predictor

sh ./run.sh

date
)
