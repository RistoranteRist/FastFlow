# FastFlow
An unofficial PyTorch implementation of [*FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows*](https://arxiv.org/abs/2111.07677) (Jiawei Yu et al.).  


We modified some of [FrEIA](https://github.com/VLL-HD/FrEIA) module to output Jacobian determinant which has same shape of the input data, [here](https://github.com/RistoranteRist/FastFlow/blob/main/model.py#L42-L336).  

## Installation

1. Clone this repository.  
2. Download MVTecAD dataset from https://www.mvtec.com/company/research/datasets/mvtec-ad and place it in the directory of your choice.  
3. Install python packages on your system with `pip install -r requirements.txt`.  

Versions of our system is listed below.

```
OS      : Ubuntu 18.04.5
CUDA    : 11.3
cudnn   : 8.2.0.53-1
python  : 3.7.11
FrEIA   : 0.2
```

## Training models

1. Replace paths (and other configs if needed) in `config.py` to fit your environment.  

```
mvtec_path = "/path/to/MVtecAD" ## path you placed the dataset.
weight_path = "./weights" ## directory to save fastflow model weights.
result_path = "./results" ## directory to save logs.
```

2. Run `python main.py`.  

Evaluation on test dataset runs every `validate_per_epoch`(in config.py) epochs.  

## Metrics

Image level AUROC  

| category |  bottle  |  cable  |  capsule  |  carpet  |  grid  |  hazelnut  |  leather  |  metul_nut  |  pill  |  screw  |  tile  |  toothbrush  |  transistor  |  wood  |  zipper  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| impl |  1.000 |  0.919  |  0.977  |  1.000  |  0.998  |  1.000  |  1.000  |  0.998  |  0.992  |  0.846  |  0.999  |  0.872  |  0.965  |  0.987  |  0.942  |
| paper |  1.000  |  1.000 |  1.000  |  1.000  |  0.997  |  1.000  |  1.000  |  1.000  |  0.994  |  0.978  |  1.000  |  0.944  |  0.998  |  1.000  |  0.995  |
| diff |  0.000  |  -0.081 |  -0.023  |  0.000  |  0.001  |  0.000  |  0.000  |  -0.002  |  -0.002  |  -0.132  |  -0.001  | -0.072  |  -0.033  |  -0.013  |  -0.053  |

Pixel Level AUROC

| category |  bottle  |  cable  |  capsule  |  carpet  |  grid  |  hazelnut  |  leather  |  metul_nut  |  pill  |  screw  |  tile  |  toothbrush  |  transistor  |  wood  |  zipper  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| impl |  0.983 |  0.977  |  0.991  |  0.995  |  0.978  |  0.991  |  0.995  |  0.980  |  0.989  |  0.992  |  0.966  |  0.987  |  0.944  |  0.959  |  0.978  |
| paper |  0.977  |  0.984 |  0.991  |  0.994  |  0.983  |  0.991  |  0.995  |  0.985  |  0.992  |  0.994  |  0.963  |  0.989  |  0.973  |  0.970  |  0.987  |
| diff |  0.006  |  -0.007 |  0.000  |  0.001  |  -0.005  |  0.000  |  0.000  |  -0.005  |  -0.003  |  -0.002  |  0.003  | -0.002  |  -0.029  |  -0.011  |  -0.009  |