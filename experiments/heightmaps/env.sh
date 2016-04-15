#!/bin/bash

export HOSTNAME=`hostname`
export DATA_DIR=/Volumes/CB_RESEARCH/heightmaps
if [ $HOSTNAME == "cuda4.rdgi.polymtl.ca" ]; then
  export DATA_DIR=/storeSSD/cbeckham/heightmaps
elif [ $HOSTNAME == "cuda1" ]; then
  export DATA_DIR=~/heightmaps
fi
