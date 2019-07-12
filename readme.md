# RRNet - Re Regression Net for Object Detection in Drones Image

*Geo, Yu Zhang, Qingxuan LV, Shuo Wei, andd Xiaorui Wang*

This repo is created for ICCV Drones Detection 2019. We proposed RRNet, which is an anchor-free detector with a re-regression module.

## Architecture

![](./assets/arch.png)

## Demo

![](./assets/results.png)

## Environment

- PyTorch 1.1.0
- python3+

## Setup

```bash
cd ext/nms && make && cd ..

# Prepare Drones dataset in 'data' folder.

# For Training RRNet
python3 scripts/RRNet/train.py

# For evaluation
python3 scripts/RRNet/eval.py
```
