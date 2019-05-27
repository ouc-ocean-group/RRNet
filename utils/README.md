# ICCVDataSet

2019.05.12

## Files introduction

|filename|introduction|
|:----|:-----:|
|model_tools.py|the Class Dataset for ICCV|

## Detail

### model_tools.py

#### FunctionName: get_backbone

##### Params

- **bb_name** : backbone name
- **pretrained** : if us pretrained weights 

##### Return

Object of BackBone Module

#### FunctionName: get_fpn

##### Params

- **fp_name** : FPN net name

##### Return

Object of FPN Module

#### FunctionName: get_detector

##### Params

- **det_name** : detector name
- **planes**: channels for output

##### Return

Object of detector Module


#### FunctionName: get_trident

##### Params

- **backbone** : backbone name  `['trires50', 'trires101']`
- **deform**: bool `True or False`

##### Return

Trident ResNetV2 Backbone with/without deformable convolution
