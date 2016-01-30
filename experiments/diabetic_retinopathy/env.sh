#!/bin/bash

if [ $HOSTNAME == "chris" ]; then
    export DATA_DIR=/Volumes/CB_RESEARCH/dr-data/
elif [ $HOSTNAME == "cuda4.rdgi.polymtl.ca" ]; then
    export DATA_DIR=/storeSSD/cbeckham/dr-data/
fi
