#!/usr/bin/env bash
# Evaluation scripts for Center Net.

# Set pretrained model path
PRETRAINED_MODEL=zhangyu/25e-42xUPHWConv/CenterNet/ckp-99999.pth

# Prepare for data.
cd data
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/DronesDET.tar
tar -I pigz -xf DronesDET.tar
cd ..

# Download pretrained model.
mkdir log && cd log
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/$PRETRAINED_MODEL
cd ..
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/geo/hourglass.pth

# Make ext module.
cd ext/nms
make
cd ../..

# Start evaluating.
mkdir result
python3 eval_ctnet.py
