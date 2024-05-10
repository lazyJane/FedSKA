# Unsupervised Graph Structure-Assisted Personalized Federated Learning

Official codes for ECAI '23 full paper: Unsupervised Graph Structure-Assisted Personalized Federated Learning

<p align="center">
  <img width="" height="400" src=./fig/overall.png>
</p>


## Data Preparation
We provide four federated benchmark datasets spanning a wide range of machine learning tasks: handwritten character recognition(MNIST), image classification(CIFRA10), language modeling (Shakespeare) and traffic forecasting(METR-LA).

Shakespeare dataset is naturally non-i.i.d distributed where each client represents a character. For MNIST and CIFAR-10 datasets, we artificially partitioned the raw dataset using a parameter q (shards) to control the level of heterogeneity.
METR-LA [35] is a traffic dataset that has a graph topology connecting sensors on roads. Each sensor on the road can be considered a client in the federated learning system, contributing data collected from real-world sources with a non-IID distribution.

The following table summarizes the datasets and models

|Dataset         | Task |  Model |
| ------------------  |  ------|------- |
| MNIST   |     Handwritten character recognition       |     2-layer CNN + 2-layer FFN  |
| CIFAR10   |     Image classification        |      MobileNet-v2 |
| Shakespeare |     Next character prediction        |      Stacked LSTM    |
| METR-LA |     Traffic forecasting        |      GRU    |

See the README.md files of respective dataset, i.e. data/ for instructions on generating data.


## Run experiments on the *Mnist* Dataset:
```
nohup python -u main.py --dataset Mnist-allocation_shards5-ratio1.0-u100 --algorithm FedSKA\
--batch_size 32 --tau 0.9 --k 30 --feature_hidden_size 64 --beta 0.01 --num_users 100   --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 4 > ./acc_loss_record/mnist/E=1/r=1.0/FedSKA.out 2>&1 &
```
We provide example scripts to run paper experiments under experiments/ directory.

----
