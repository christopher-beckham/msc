#!/bin/bash

core_dir=aligned
new_dir=aligned_256x256
mkdir ${new_dir}
for folder in `ls ${core_dir}`; do
  for filename in `ls ${core_dir}/$folder/*.jpg`; do
    filename_base=`basename $filename`
    echo $folder ${filename_base}
    mkdir ${new_dir}/$folder
    convert ${core_dir}/$folder/${filename_base} -resize 256x256 ${new_dir}/$folder/${filename_base}
  done
done
