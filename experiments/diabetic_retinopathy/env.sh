#!/bin/bash

if [ $HOSTNAME == "chris" ]; then
    export DATA_DIR=/Volumes/CB_RESEARCH/dr-data/
elif [ $HOSTNAME == "cuda4.rdgi.polymtl.ca" ]; then
    export DATA_DIR=/storeSSD/cbeckham/dr-data/
elif [ $HOSTNAME == "bart16" ]; then
    export DATA_DIR=/data/lisa/data/kaggle_dr/train-trim-ben-r400-512/
fi
