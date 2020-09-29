# Kraken

This is the opensource warehouse for Kraken's sparsity aware training algorithm and `RAdaGrad` optimizer implemented with `pytorch.`
We also provide the code of baseline TensorFlow for reproduction.

## Install

- install torch==1.4

- install tensorflow1

- install `radagrad.py` and `adagrad.py` to torch

  ```
  mv sparsity-aware-training/radagrad.py sparsity-aware-training/adagrad.py sparsity-aware-training/__init__.py YOUR_PATH_OF_`site-packages/torch/optim`
  ```



## Use

Please check out  `main.py` to see how to use it.

```
python3 sparsity-aware-training/main.py -h

or


python3 tf-baseline/main.py -h
```



## Acknowledgement

High tribute shall be paid to [neo.jia.lin](https://github.com/neolinsu) for his contribution to this repository.


