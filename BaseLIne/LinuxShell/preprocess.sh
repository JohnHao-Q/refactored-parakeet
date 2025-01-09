#!/usr/bin/sh

data_src=${1-'/home/samidare/Kitti'}
data_dest=${2-'/home/samidare/Datasets/sampled_500'}
num_samples=${3-4633}

mkdir -p $data_dest

python3 Datasets/Kitti_loader.py --num_samples $num_samples --datapath $data_src --dest $data_dest

# copy non existent files over (ground truth etc)
cp -r -n $data_src/* $data_dest
