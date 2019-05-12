# ICCVDataSet

2019.05.05

## Files introduction

|filename|introduction|
|:----|:-----:|
|dataset.py|the Class Dataset for ICCV|

## Detail

### dataset.py

ClassName: DronesDET()


Function  
- <font color='blue' >\_\_Init\_\_() </font>
  - **Params** 
    - <font color='green' >root_dir</font> Root path of images and annotations dir
- <font color='blue' >\_\_len\_\_() </font>
  - **Return**: Num of datasets
- <font color='blue' >\_\_getitem\_\_() </font>
  - **Params**: 
    - <font color='green' >item</font> index of item 
  - **Return**: Tuple(image[PIL image],annotation[list]) P.S. annotation is list like
    \[[1,1,1,1,1],[1,1,1,1,1]]