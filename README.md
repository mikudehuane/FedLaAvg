# Federated-Latest-Averaging
Implementation of https://arxiv.org/abs/2002.07399

## Algorithm illustration
![Algorithm Overview](figure/overview.png)

## Usage
- Please download [initialize.sh](initialize.sh) into an empty directory, and run it via the 
    following command to initialize the project 
    (type the commands in your shell rather than copy to a shell script please).
```shell script
# clone project
mkdir FedLaAvg
cd FedLaAvg
git clone https://github.com/mikudehuane/FedLaAvg scripts

# install packages
conda create -n FedLaAvg python==3.6.9
conda activate FedLaAvg
pip install -r scripts/requirements.txt

# configuring project
echo -e "use_cuda = False\nproject_dir = \"$(pwd)\"" > scripts/config.py

# generating initial model
mkdir cache data
python scripts/train/model.py

# download glove embedding
mkdir -p models/glove/
wget -P models/glove/ http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip -d models/glove/ models/glove/glove.twitter.27B.zip

# download sentiment dataset
mkdir -p raw_data/Sentiment140
wget -P raw_data/Sentiment140/ http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip -d raw_data/Sentiment140/ raw_data/Sentiment140/trainingandtestdata.zip
```
- Then, please run experiments via train/server.py. Enter "python train/server.py -h" to see the help. 

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

# \beta=0.05
python server.py -ds mnist -alg lastavg -N 1000 -be 0.05 -C 10 -E 100 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# all available clients participate
python server.py -ds mnist -alg lastavg -N 1000 -be 1.0 -C 10 -E 100 --alpha 0.1 --num_rounds 10000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# N=200
python server.py -ds mnist -alg lastavg -N 200 -be 0.1 -C 10 -E 100 --alpha 0.1 --num_rounds 4000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# N=600
python server.py -ds mnist -alg lastavg -N 600 -be 0.1 -C 10 -E 100 --alpha 0.1 --num_rounds 4000 --lr_strategy const --init_lr 0.01 -nm 0.0 -ns 1.0 -stg number

# C=1
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


### Sentiment140 FedAvg
```shell script
# E=24 \alpha=0
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 10 -E 24 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1

# E=120 \alpha=0
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1

# E=120 \alpha=0.25
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --fv_min 0.25 --fv_max 0.75

# E=120 \alpha=0.5
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --fv_min 0.5 --fv_max 0.5

# E=240 \alpha=0
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 10 -E 240 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1
```

### Sentiment140 FedLaAvg
```shell script
# E=24 \alpha=0
python server.py -ds sentiment140 -alg lastavg -N 1000 -be 0.1 -C 10 -E 24 --lr_strategy const -lr 0.01 --num_rounds 30000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1

# E=120 \alpha=0
python server.py -ds sentiment140 -alg lastavg -N 1000 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1

# E=120 \alpha=0.25
python server.py -ds sentiment140 -alg lastavg -N 1000 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --fv_min 0.25 --fv_max 0.75

# E=120 \alpha=0.5
python server.py -ds sentiment140 -alg lastavg -N 1000 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --fv_min 0.5 --fv_max 0.5

# E=240 \alpha=0
python server.py -ds sentiment140 -alg lastavg -N 1000 -be 0.1 -C 10 -E 240 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1

# \beta=0.2
python server.py -ds sentiment140 -alg lastavg -N 1000 -be 0.2 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1

# all available clients participate
python server.py -ds sentiment140 -alg lastavg -N 1000 -be 1.0 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1

# N=200
python server.py -ds sentiment140 -alg lastavg -N 200 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1

# N=600
python server.py -ds sentiment140 -alg lastavg -N 200 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1

# C=1
python server.py -ds sentiment140 -alg lastavg -N 1000 -be 0.1 -C 1 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1
```

### Sentiment140 FedSGD
```shell script
# E=24 \alpha=0
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 1 -E 24 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1

# E=120 \alpha=0
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 1 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1

# E=120 \alpha=0.25
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 1 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --fv_min 0.25 --fv_max 0.75

# E=120 \alpha=0.5
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 1 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --fv_min 0.5 --fv_max 0.5

# E=240 \alpha=0
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 1 -E 240 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1
```


### Sentiment140 FedProx
```shell script
# E=24 \alpha=0
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 10 -E 24 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 -mu 0.01

# E=120 \alpha=0
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1  -mu 0.01

# E=120 \alpha=0.25
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --fv_min 0.25 --fv_max 0.75 -mu 0.01

# E=120 \alpha=0.5
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --fv_min 0.5 --fv_max 0.5 -mu 0.01

# E=240 \alpha=0
python server.py -ds sentiment140 -alg fedavg -N 1000 -be 0.1 -C 10 -E 240 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 -mu 0.01
```

### Sentiment140 FedWaitAvg
```shell script
# E=24 \alpha=0
python server.py -ds sentiment140 -alg waitavg -N 1000 -be 0.1 -C 10 -E 24 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --tb_every=200 --checkpoint_every=1000

# E=120 \alpha=0
python server.py -ds sentiment140 -alg waitavg -N 1000 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --tb_every=200 --checkpoint_every=1000

# E=120 \alpha=0.25
python server.py -ds sentiment140 -alg waitavg -N 1000 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --fv_min 0.25 --fv_max 0.75 --tb_every=200 --checkpoint_every=1000

# E=120 \alpha=0.5
python server.py -ds sentiment140 -alg waitavg -N 1000 -be 0.1 -C 10 -E 120 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --fv_min 0.5 --fv_max 0.5 --tb_every=200 --checkpoint_every=1000

# E=240 \alpha=0
python server.py -ds sentiment140 -alg waitavg -N 1000 -be 0.1 -C 10 -E 240 --lr_strategy const -lr 0.01 --num_rounds 50000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --tb_every=200 --checkpoint_every=1000
```

### Sentiment140 Sequential SGD
```shell script
python server.py -ds sentiment140 -alg sgd -N 1000 -be 0.001 -C 10 -E 240 --lr_strategy const -lr 0.01 --num_rounds 30000 --model_ori lstm_i25_h16 --glove_model glove.twitter.27B.25d.txt -am modeled_mid -fv 1 --batch_size 200
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
