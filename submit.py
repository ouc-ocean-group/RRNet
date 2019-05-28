from pypai import PAI

pai = PAI(username='zhangyu', passwd='sdojj8689075')



#zhangyu:1.3
pai.submit()

#train:     /bin/bash /root/mount.sh && cd $PAI_JOB_NAME && cd libs && sh build.sh && python3 build.py && cd ../cc_attention && sh build.sh && python3 build.py && cd .. && python3 train.py --random-mirror --random-scale --restore-from /root/data/models/zhangyu/resnet101-imagenet.pth --gpu 0,1 --learning-rate 1e-2 --input-size 640,640 --weight-decay 1e-4 --batch-size 4 --num-steps 150000 --recurrence 2

#test:      /bin/bash /root/mount.sh && cd $PAI_JOB_NAME && cd libs && sh build.sh && python3 build.py && cd ../cc_attention && sh build.sh && python3 build.py && cd .. && python3 evaluate.py --restore-from /root/data/models/zhangyu/CS_scenes_60000.pth --gpu 0 --recurrence 2

#pytorch:v0.4.0


#2080test   /bin/bash /root/mount.sh && cd $PAI_JOB_NAME && python3 evaluate.py --restore-from /root/data/models/zhangyu/CS_scenes_60000.pth --gpu 0 --recurrence 2

#/bin/bash /root/mount.sh && cd $PAI_JOB_NAME && cd experiments && python3 train.py


# /bin/bash /root/mount.sh && cd $PAI_JOB_NAME && cd experiments && python3 evaluate.py

#cd $PAI_JOB_NAME && hdfs dfs -get $PAI_DATA_DIR && tar -xf cityscapes.tar && cd experiments && python3 train.py

# tensorboard --port *** --logdir ./