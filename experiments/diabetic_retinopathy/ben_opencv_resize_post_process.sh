#!/bin/bash

if [ -z $DATA_DIR ]; then
    echo "$DATA_DIR does not exist - source env.sh"
    exit 1
fi

if [ -z $1 ]; then
    echo "Specify a source folder (e.g. train, test)"
    exit 1
fi

if [ -z $2 ]; then
    echo "Specify a folder to dump the converted images in"
    exit 1
fi

if [ -z $3 ]; then
    echo "Specify an image size"
    exit 1
fi

export TMP_SRC_DIR=$1
export TMP_NEW_DIR=$2
export IMG_SIZE=$3

if [ ! -d $DATA_DIR/$TMP_NEW_DIR ]; then

    mkdir $DATA_DIR/$TMP_NEW_DIR

    cd $DATA_DIR/$TMP_SRC_DIR
    find . | grep .jpeg | parallel 'convert $DATA_DIR/$TMP_SRC_DIR/{} -fuzz 3% -trim -resize $IMG_SIZE -background "rgb(127,127,127)" -gravity center -extent $IMG_SIZE $DATA_DIR/$TMP_NEW_DIR/{}'

fi
