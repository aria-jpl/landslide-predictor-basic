#!/bin/bash

set -e

name=landslide-predictor-basic

docker run \
    --name ${name} \
    -v /home/xing/hysds:/home/xing/hysds \
    -v /home/xing/hysds/landslide/try:/home/ops/try \
    --rm \
    -it \
    hysds/${name}:20200327 bash
