mkdir /root/.cache/torch
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/DronesDET.tar
tar -xf DronesDET.tar
cp -r ./DronesDET ./data
ln -s /root/.torch/models /root/.cache/torch/checkpoints
