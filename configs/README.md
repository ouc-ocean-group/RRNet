# ICCVDataSet

2019.05.12

## Files introduction

|filename|introduction|
|:----|:-----:|
|demo_config.py|the global config for ICCV|

## Detail

### demo_config.py

ClassName: Config (static class)


#### Object
- Model

|name|introduction|
|:-----|:----|
|pretrained|if use pretrained weights|
|backbone|name for backbone like `resnet50`|
|fpn|name for backbone like `fpn`|
|cls_detector|name for class detector like `retinanet_detector`|
|loc_detector|name for location detector like `retinanet_detector`|
|num_anchors|number of anchors|
|num_classes|number of classes|