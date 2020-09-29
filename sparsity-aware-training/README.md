# Kraken

This is the opensource warehouse for Kraken's sparsity aware training algorithm and `RAdaGrad` optimizer  implemented with `pytorch.`

## Install

- install torch==1.4

- install tensorflow

- install `radagrad.py` and `adagrad.py` to torch

  ```
  mv radagrad.py adagrad.py __init__.py YOUR_PATH_OF_`site-packages/torch/optim`
  ```



## Use

Please check out  `main.py` to see how to use it.

```
python3 main.py -h
```



## Acknowledgement

High tribute shall be paid to [neo.jia.lin](https://github.com/neolinsu) for his contribution to this repository.


