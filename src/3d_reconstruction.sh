#!/bin/bash

cd colmap
# run colmap
bash rename.sh ../../data/$1/images
bash run_colmap.sh ../../data/$1

cd ../masking
# perform masking
python mask.py --data_path ../../data/$1/
python apply_mask.py --data_path ../../data/$1/

cd ../colmap
# mask the .ply files
bash masked_pcl.sh ../../data/$1
