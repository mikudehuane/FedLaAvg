# Federated-Latest-Averaging
Implementation of https://arxiv.org/abs/2002.07399

## Algorithm illustration
![Algorithm Overview](figure/overview.png)

## Usage
Please run experiments via train/server.py. 
Enter "python train/server.py -h" to see the help. 

## Dependency
+ beautifulsoup4 (4.9.1)
+ bs4 (0.0.1)
+ gensim (3.8.3)
+ lxml (4.5.2)
+ matplotlib (3.2.1)
+ nltk (3.5)
+ numpy (1.18.2)
+ Pillow (7.1.1)
+ tensorboard (2.2.2)
+ torch (1.4.0)
+ torchvision (0.5.0)


## Instructions to Reproduce the Results

### MNIST FedAvg
```shell script
# E=50 D=1
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 10 -E 50 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=100 D=1
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 10 -E 100 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=100 D=3
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 10 -E 100 --alpha 0.3 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=100 D=5
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 10 -E 100 --alpha 0.5 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=200 D=1
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 10 -E 200 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number
```

### MNIST FedLaAvg
```shell script
# E=50 D=1
python server.py -ds mnist -alg lastavg -N 1000 -be 0.1 -C 10 -E 50 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=100 D=1
python server.py -ds mnist -alg lastavg -N 1000 -be 0.1 -C 10 -E 100 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=100 D=3
python server.py -ds mnist -alg lastavg -N 1000 -be 0.1 -C 10 -E 100 --alpha 0.3 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=100 D=5
python server.py -ds mnist -alg lastavg -N 1000 -be 0.1 -C 10 -E 100 --alpha 0.5 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=200 D=1
python server.py -ds mnist -alg lastavg -N 1000 -be 0.1 -C 10 -E 200 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# \beta = 0.05
python server.py -ds mnist -alg lastavg -N 1000 -be 0.05 -C 10 -E 100 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# all available clients participate
python server.py -ds mnist -alg lastavg -N 1000 -be 1.0 -C 10 -E 100 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# N = 200
python server.py -ds mnist -alg lastavg -N 200 -be 0.1 -C 10 -E 100 --alpha 0.1 --num_rounds 4000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# N = 600
python server.py -ds mnist -alg lastavg -N 600 -be 0.1 -C 10 -E 100 --alpha 0.1 --num_rounds 4000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# C = 1
python server.py -ds mnist -alg lastavg -N 1000 -be 0.1 -C 1 -E 100 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number
```

### MNIST FedProx
```shell script
# E=50 D=1
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 10 -E 50 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number -mu 1.0

# E=100 D=1
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 10 -E 100 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number -mu 1.0

# E=100 D=3
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 10 -E 100 --alpha 0.3 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number -mu 1.0

# E=100 D=5
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 10 -E 100 --alpha 0.5 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number -mu 1.0

# E=200 D=1
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 10 -E 200 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number -mu 1.0
```

### MNIST FedSGD
```shell script
# E=50 D=1
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 1 -E 50 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=100 D=1
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 1 -E 100 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=100 D=3
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 1 -E 100 --alpha 0.3 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=100 D=5
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 1 -E 100 --alpha 0.5 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=200 D=1
python server.py -ds mnist -alg fedavg -N 1000 -be 0.1 -C 1 -E 200 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number
```

### MNIST FedWaitAvg
```shell script
# E=50 D=1
python server.py -ds mnist -alg waitavg -N 1000 -be 0.1 -C 10 -E 50 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=100 D=1
python server.py -ds mnist -alg waitavg -N 1000 -be 0.1 -C 10 -E 100 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=100 D=3
python server.py -ds mnist -alg waitavg -N 1000 -be 0.1 -C 10 -E 100 --alpha 0.3 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=100 D=5
python server.py -ds mnist -alg waitavg -N 1000 -be 0.1 -C 10 -E 100 --alpha 0.5 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# E=200 D=1
python server.py -ds mnist -alg waitavg -N 1000 -be 0.1 -C 10 -E 200 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number
```


### MNIST Sequential SGD
```shell script
python server.py -ds mnist -alg sgd -N 10 -be 0.1 -C 10 -E 10 --alpha 0.5 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -te 5 --batch_size 500
```


## Citation
```
@article{DBLP:journals/corr/abs-2002-07399,
  author    = {Yikai Yan and
               Chaoyue Niu and
               Yucheng Ding and
               Zhenzhe Zheng and
               Fan Wu and
               Guihai Chen and
               Shaojie Tang and
               Zhihua Wu},
  title     = {Distributed Non-Convex Optimization with Sublinear Speedup under Intermittent
               Client Availability},
  journal   = {CoRR},
  volume    = {abs/2002.07399},
  year      = {2020}
}
```
