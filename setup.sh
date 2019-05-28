cd data
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/DronesDET.tar
tar -I pigz -xf DronesDET.tar
cd ..
python3 train_centernet.py
cd log
mv Cen* CenterNet
hdfs dfs -mkdir -p $PAI_DEFAULT_FS_URI/data/models/zhangyu/20190527/
hdfs dfs -mkdir -p $PAI_DEFAULT_FS_URI/data/models/zhangyu/20190527/
