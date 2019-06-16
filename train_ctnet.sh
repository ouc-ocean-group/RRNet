#!/usr/bin/env bash
# Training scripts for Center Net.

# Prepare for data.
cd data
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/DronesDET.tar
tar -I pigz -xf DronesDET.tar
cd ..

# Download pretrained backbone model.
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/geo/hourglass.pth

# Make ext module.
cd ext/nms
make
cd ../..

# Start training.
python3 train_ctnet.py

# Save checkpoints and logs.
cd log
hdfs dfs -mkdir -p $PAI_DEFAULT_FS_URI/data/models/iccvdet/CenterNet/
for f in *; do
    echo "Saving Checkpoint and logs... -> $f"
    hdfs dfs -put -f $f $PAI_DEFAULT_FS_URI/data/models/iccvdet/CenterNet/$f
done

# Evaluation.
mkdir result
python3 eval_ctnet.py