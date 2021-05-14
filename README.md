# Federated-Latest-Averaging
Implementation of https://arxiv.org/abs/2002.07399

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

## Usage
Please run experiments via train/server.py. Enter "cd train && python server.py -h" to see the help. Running parameters are discribed as follows (running argument may have changed, please refer to the output of "python server.py -h" when running the code).

 
## Algorithm illustration and some experiment results
### Algorithm overview 
<p align="center">
  <img src="figure/overview.png" width = "80%" height = "80%" alt="Algorithm Overview" align=center />
</p>

### Availability setting
<p align="center">
  <img src="figure/gantt.png" width = "80%" height = "80%" alt="Availability Setting" align=center />
</p>

### Comparison between FedAvg and FedLaAvg
<p align="center">
  <img src="figure/comp_main.png" width = "50%" height = "50%" alt="Comparison between FedAvg and FedLaAvg" align=center />
</p>

### Comparison among different $N$, $E$, and $\beta$
<p align="center">
  <img src="figure/N.png" width = "50%" height = "50%" alt="Comparison among different $N$" align=center />
</p>
<p align="center">
  <img src="figure/E.png" width = "50%" height = "50%" alt="Comparison among different $E$" align=center />
</p>
<p align="center">
  <img src="figure/beta.png" width = "50%" height = "50%" alt="Comparison among different $\beta$" align=center />
</p>

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
