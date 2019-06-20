#!/usr/bin/env bash
# Training scripts for Center Net.

# Prepare for data.
echo "=> Preparing data..."
cd data
echo "   Downloading data..."
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/DronesDET.tar

echo "   Decompress data..."
tar -I pigz -xf DronesDET.tar
cd ..

# Download pretrained backbone model.
echo "=> Downloading pretrained model..."
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/geo/hourglass.pth

# Make ext module.
echo "=> Build extra module..."
cd ext/nms
make > /dev/null 2>&1
cd ../..

# Start training.

echo "=> Start training..."
if [[ $1 = "normal" ]]; then
    python3 train_ctnet.py
elif [[ $1 = "kl" ]]; then
    python3 train_ctnet_kl.py
else
    echo "Wrong model!"
fi

# Save checkpoints and logs.
echo "=> Saving checkpoints and logs..."
cd log
hdfs dfs -mkdir -p $PAI_DEFAULT_FS_URI/data/models/iccvdet/CenterNet/
for f in *; do
    echo "   $f"
    hdfs dfs -put -f $f $PAI_DEFAULT_FS_URI/data/models/iccvdet/CenterNet/$f
    mv $f ../
done
cd ..


# Evaluation.
echo "=> Start evaluating..."
if [[ $1 = "normal" ]]; then
    python3 eval_ctnet.py
elif [[ $1 = "kl" ]]; then
    python3 eval_ctnet_kl.py
else
    echo "Wrong model!"
fi

echo "=> All done."