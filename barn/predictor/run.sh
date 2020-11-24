#!/bin/bash

set -e

(
eval "$(/opt/conda/bin/conda shell.bash hook)"

cd ./pred_ps_package-2/ps_app

dirPath=`pwd`

# ref: from giangi
# python3 predict_ps.py -d ~/Volumes/giangidt/pred_mintpy/cali/pred_in -i model_ps_20190814.json -p ~/Volumes/giangidt/pred_mintpy/cali/pred_out/ -m model_20190814_0 -g -1 -b 16384 -s _data.pck -r ~/Workspace/Data/ml_log/
#conda run -n python368 python predict_ps.py -d $dirPath/../../input -i model_ps_20190814.json -p $dirPath/../../output -m model_20190814_0 -g -1 -b 16384 -s pickle -r ../model
conda run -n python368 python predict_ps.py -d $dirPath/../../input -i model_ps_20190814.json -p $dirPath/../../output -m model_20190814_0 -g -1 -b 16384 -s pck -r ../model

)

