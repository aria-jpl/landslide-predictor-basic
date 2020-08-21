#!/bin/bash

#export LD_LIBRARY_PATH=/opt/conda/lib:/usr/lib:/usr/lib64:/usr/local/lib:$LD_LIBRARY_PATH
#export PATH=/opt/conda/bin:$PATH

# https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html

whoami

echo $SHELL

head ~/.bashrc

echo ${EUID:-0}

echo $(id -u)

cat /etc/passwd | grep ops

# when built by docker, this is not sourced by bash, do this explicitly here.
# wirte something about we are using hysds/pge-base:latest as docker base and it already has conda
# and we resue this conda.
source ~/.bashrc

which conda

conda create --name python368 python=3.6.8 anaconda -y

#conda init bash

#source ./conda-init.sh

which python

conda activate python368

which python

pip install -r ./requirements.txt
