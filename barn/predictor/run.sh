#!/bin/bash

set -e

(
eval "$(/opt/conda/bin/conda shell.bash hook)"

cd ./pred_ps_package-2/ps_app

conda run -n python368 python predict_ps.py -d ~/predictor/input -i model_ps_20190814.json -p ~/predictor/output -m model_20190814_0 -g -1 -b 16384 -s pickle -r ../model

)

