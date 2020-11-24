#!/bin/bash -xv

#set -e

# ------------------------------------------------------------------------------
# Automatically determines the full canonical path of where this script is
# located--regardless of what path this script is called from. (if available)
# ${BASH_SOURCE} works in both sourcing and execing the bash script.
# ${0} only works for when execing the bash script. ${0}==bash when sourcing.
BASE_PATH=$(dirname "${BASH_SOURCE}")
# convert potentially relative path to the full canonical path
BASE_PATH=$(cd "${BASE_PATH}"; pwd)
# get the name of the script
BASE_NAME=$(basename "${BASH_SOURCE}")
# ------------------------------------------------------------------------------

WORK_DIR=$(pwd)
echo "WORK_DIR:" $WORK_DIR


s3Uri=`python $BASE_PATH/get_param.py`


eval "$(/opt/conda/bin/conda shell.bash hook)"

echo "which conda"
which conda

#------
# prepare timeseries
(
echo "prepare timeseries"

date

cd $BASE_PATH/ts_mintpy

#s3Uri=s3://aria-dev-lts-fwd-xing/test-landslide/from-hook-20200820/_mintpy_time_series

sh -xv ./run.sh $s3Uri

# move some files under ./output to input dir of predictor
#cp -pr ./output/pred_input_*_data.pickle ${BASE_PATH}/predictor/input
cp -pr ./output/pred_input_*_data.pck ${BASE_PATH}/predictor/input

date
)

#------
# run predictor
(
echo "run predictor"

date

cd $BASE_PATH/predictor

sh -xv ./run.sh

date
)

#------
# run plot
(
echo "run plot"

date

cd ${BASE_PATH}/plot

sh -xv ./run.sh

date
)

#------
# save result
(
echo "create dataset"

date

cd ${BASE_PATH}

sh -xv ./create_dataset.sh $WORK_DIR $BASE_PATH

date
)
