#!/bin/bash

set -e

date

dirPath=/home/ops/predictor

(
cd $dirPath

tar zxvf ./pred_ps_package-2.tar.gz

mv ./TFManager.py ./pred_ps_package-2/ps_app

mv ./checkpoint ./pred_ps_package-2/model/ps_model_20190814_0

#mv ./run.sh ./pred_ps_package-2

mv ./ps.sample.pickle ./input
)
