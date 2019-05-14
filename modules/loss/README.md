# ICCVLoss

2019.05.14

## Files introduction

|filename|introduction|
|:----|:-----:|
|[focalloss.py](#focalloss_py)|the focal loss class for RetinaNet|
| [functional.py](#functional_py)|save all loss functional|

## Detail

### focalloss.py {#focalloss_py}

#### ClassName: FocalLoss() {#focalloss}

##### Introduction

This is Focalloss for RetinaNet

$FL= - \alpha_{t} * \left( 1-P_{t} \right)^{\gamma}*CE(P_{t}) $

##### Function

- **\_\_init\_\_()**
  - **params**:
    - **alpha** : a float32 number for hyperparameter
    - **gamma** : a float32 number for hyperparameter
    - **classnum** : a int number for amount of classes

### functional.py  {#functional_py}

|function name| introduction |
|:----:|:-----|
|[focal_loss](#focal_loss)| calc focal loss|
|[one_hot](#one_hot)|one-hot encoding| 

#### FunctionName: focal_loss {#focal_loss}

see [FocalLoss](#focalloss)

and it can remove target items where label==-1, use `softmax` instead of `sigmoid`

#####　Params

- **inputs** : $[N,H*W*anchers,classes]$
- **target** : $[N,H*W*anchers]$
- **alpha** : float32 number for hyperparameter
- **gamma** : float32 number for hyperparameter
- **num_classes** : int number for classes amount  

##### Return

A tensor for Focal Loss not mean

#### FunctionName: one_hot {#one_hot}

change $[A]$ to $[A,classes]$

##### Params

- **inputs** : $[A]$ longtensor or inttensor
- **num_classes** : int number for classes amount

##### Return

A tensor $[A,classes]$


    