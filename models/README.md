# ICCVModel

2019.05.12

## Files introduction

|filename|introduction|
|:----|:-----:|
|retinanet.py|the retinanet for ICCV|
|centernet.py|the centernet for ICCV|

## Detail

### retinanet.py

#### ClassName: RetinaNet()

##### Introduction

This is main retinaNet, and it has 4 parts
- backbone
- fpn
- cls_detector
- loc_detector

#### FunctionName: build_net:

##### Params
- **cfg** use  cfg.model

### centernet.py

#### ClassName: CenterNet()

##### Introduction

This is CenterNet, and it has 4 parts
- backbone
- hm_detector
- wh_detector
- reg_detector(offset)

##### Params
- **cfg** use  cfg.model

    