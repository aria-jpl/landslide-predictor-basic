#!/bin/bash

BASE_PATH=$(dirname "${BASH_SOURCE}")
BASE_PATH=$(cd "${BASE_PATH}"; pwd)
echo "BASE_PATH:" $BASE_PATH

#------
# 20200823, xing
# make a new copy of involved tools/dirs/files to avoid permission error,
# due to difference between uid built in docker image and uid used in running docker container.

mkdir -p $BASE_PATH/work

cp -avi $BASE_PATH/run.sh $BASE_PATH/work
cp -avi $BASE_PATH/get_param.py $BASE_PATH/work
cp -avi $BASE_PATH/create_dataset.sh $BASE_PATH/work

cp -avi $BASE_PATH/{ts_mintpy,predictor,plot} $BASE_PATH/work

#------
echo "pwd"
pwd

sh -xv $BASE_PATH/work/run.sh
