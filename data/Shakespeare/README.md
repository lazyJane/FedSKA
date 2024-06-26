# Shakespeare Dataset

## Introduction
This dataset is derived from the
[Leaf repository](https://github.com/TalwalkarLab/leaf) pre-processing of the
Shakespeare dataset. This dataset is built from 
*The Complete Works of William Shakespeare*. Each speaking role in each play is
considered a different device. 

Details about  LEAF were published in
"LEAF: A Benchmark for Federated Settings" [https://arxiv.org/abs/1812.01097]()

## Setup Instructions

First, Run `./get_data.sh`, then run `generate_data.py` with a choice of the following arguments:

- ```--s_frac```: fraction of the dataset to be used; default=``0.3``  
- ```--tr_frac```: train set proportion for each task; default=``0.8``
- ```--val_frac```: fraction of validation set (from train set); default=`0.0`
- ```--seed``` : seed to be used before random sampling of data; default=``12345``

## Paper Experiments

In order to generate the data split, run

```
python generate_data.py \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --seed 12345    
```


