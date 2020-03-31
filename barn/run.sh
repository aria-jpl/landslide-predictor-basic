#!/bin/bash

#set -e

# prepare timeseries
(
date

cd /home/ops/timeseries

sh ./run.sh

date
)

# run predictor
(
date

cd /home/ops/predictor/pred_ps_package-2

sh ./run.sh

date
)
