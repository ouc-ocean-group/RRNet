mkdir /root/.cache/torch
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/DronesDET.tar
tar -xf DronesDET.tar
cp -r ./DronesDET ./data
ln -s /root/.torch/models /root/.cache/torch/checkpoints
python3 train_centernet.py
cd log
mv Cen* CenterNet
hdfs dfs -mkdir -p $PAI_DEFAULT_FS_URI/data/models/zhangyu/20190527/
hdfs dfs -put -f ./CenterNet $PAI_DEFAULT_FS_URI/data/models/zhangyu/20190527/
