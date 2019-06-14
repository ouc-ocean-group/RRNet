cd data
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/DronesDET.tar
tar -I pigz -xf DronesDET.tar
cd ..
mkdir log
mkdir result
cd log
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/25e-42xUPHWConv/CenterNet/ckp-99999.pth
cd ..
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/geo/hourglass.pth
cd ext
cd nms
make
cd ..
cd ..
python3 centernet_evalresult.py
