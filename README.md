# FastFlow
Unofficial PyTorch implementation of [FastFlow](https://arxiv.org/abs/2111.07677)
[FrEIA](https://github.com/VLL-HD/FrEIA) module modified to fit the Jacobi determinant shape is used

## Training models
fill in pathes in `config.py` and run `python main.py`

## metrics

Image level AUROC

| category |  bottle  |  cable  |  capsule  |  carpet  |  grid  |  hazelnut  |  leather  |  metul_nut  |  pill  |  screw  |  tile  |  toothbrush  |  transistor  |  wood  |  zipper  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| impl |  1.000 |  0.919  |  0.977  |  1.000  |  0.998  |  1.000  |  1.000  |  0.998  |  0.992  |  0.846  |  0.999  |  0.872  |  0.965  |  0.987  |  0.942  |
| paper |  1.000  |  1.000 |  1.000  |  1.000  |  0.997  |  1.000  |  1.000  |  1.000  |  0.994  |  0.978  |  1.000  |  0.944  |  0.998  |  1.000  |  0.995  |
| diff |  0.000  |  -0.011 |  -0.023  |  0.000  |  0.001  |  0.000  |  0.000  |  -0.002  |  -0.002  |  -0.126  |  -0.001  | -0.072  |  -0.033  |  -0.013  |  -0.053  |

Pixel Level AUROC

| category |  bottle  |  cable  |  capsule  |  carpet  |  grid  |  hazelnut  |  leather  |  metul_nut  |  pill  |  screw  |  tile  |  toothbrush  |  transistor  |  wood  |  zipper  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| impl |  0.983 |  0.977  |  0.991  |  0.995  |  0.978  |  0.991  |  0.995  |  0.980  |  0.989  |  0.992  |  0.966  |  0.987  |  0.944  |  0.959  |  0.978  |
| paper |  0.977  |  0.984 |  0.991  |  0.994  |  0.983  |  0.991  |  0.995  |  0.985  |  0.992  |  0.994  |  0.963  |  0.989  |  0.973  |  0.970  |  0.987  |
| diff |  0.006  |  -0.007 |  0.000  |  0.001  |  -0.005  |  0.000  |  0.000  |  -0.005  |  -0.004  |  -0.002  |  0.003  | -0.002  |  -0.029  |  -0.011  |  -0.009  |