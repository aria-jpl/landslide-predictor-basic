#!/bin/bash

set -e

#if [ "$#" -ne 1 ]; then
#    echo "Usage: $0 dummy" >&2
#    exit 1
#fi

date

inputDirPath=../ts_mintpy/input
outputDirPath=../ts_mintpy/output

# ref: from giangi
# python3 mintpy2pred.py --action create_png -o ~/Volumes/giangidt/pred_mintpy/cali/pred_in/ -d ~/Volumes/giangidt/pred_mintpy/cali/inputs  -t timeseries_demErr_1.h5 -n pred_input_filt --pdir  ~/Volumes/giangidt/pred_mintpy/cali/pred_out/ --fig_name ~/Downloads/test test_-121.887_-121.501_36.002_36.300.png

nice -20 \
    conda run -n python368 \
    python3.6 ../ts_mintpy/mintpy2pred.py \
    --action create_png \
    -d $inputDirPath \
    -o $outputDirPath \
    -t timeseries_demErr.h5 \
    -n pred_input_filt \
    --pdir ../predictor/output \
    --fig_name ./test

#nice -20 \
#    conda run -n python368 \
#    python3.6 ./mintpy_ps.py \
#    $mintpyOutputDirPath \
#    $predictorInputDirPath \
#    $predictorOutputDirPath \
#    $outputFilePath

date
