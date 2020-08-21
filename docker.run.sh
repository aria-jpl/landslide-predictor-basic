#!/bin/bash

set -e

name=landslide-predictor-basic

#    --rm \
docker run \
    --name ${name} \
    -v ~/hysds/landslide:/home/ops/landslide \
    -it \
    hysds/${name}:20200821 /bin/bash
#    --entrypoint=/bin/bash \
