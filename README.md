# Variational Latent Oracle Guiding (VLOG)

This repository includes the source code for the ICLR 2022 paper "Variational oracle guiding for reinforcement learning" (https://openreview.net/forum?id=pjqqxepwoMy) by Dongqi Han, Tadashi Kozuno, Xufang Luo, Zhao-Yun Chen, Kenji Doya, Yuqing Yang and Dongsheng Li.

```
@inproceedings{han2022variational,
    title={Variational oracle guiding for reinforcement learning},
    author={Dongqi Han and Tadashi Kozuno and Xufang Luo and Zhao-Yun Chen and Kenji Doya and Yuqing Yang and Dongsheng Li},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=pjqqxepwoMy}
}
```

# Dependence

- torch
- numpy
- scipy
- gym
- gym-maze (https://github.com/MattChanTK/gym-maze)
- minatar (https://github.com/kenjyoung/MinAtar)

For the experiments on Mahjong, see below.

# How to use

## Maze and noisy MinAtar 

For example, to run experiment of VLOG in the Maze task.
```
python run_vlog.py --env 0 --type_id 1
```
where the arguments are explained as follows

| env | Environment    |
|-----|----------------|
| 0   | Maze           |
| 1   | Breakout       |
| 2   | Seaquest       |
| 3   | Space invaders |
| 4   | Freeway        |
| 5   | Asterix        |
 
| type_id | Method         |
|---------|----------------|
| 1       | VLOG           |
| 2       | Baseline       |
| 3       | Oracle         |
| 4       | VLOG-no oracle |
| 5       | Suphx-style    |
| 6       | OPD-style      |

After finishing an experiment, the data (including performance curve etc.) and model will be saved to ./data/

To load a model, one may run
```
import torch
model = torch.load("./data/xxx.model")
```

To load the data, one may run
```
import scipy
data = scipy.io.loadmat("./data/xxx.mat")
```
to get a python dictionary data.

Then one can plot the performance curve, e.g.,
```
import matplotlib.pyplot as plt
plt.plot(data["global_steps"], data["performance_greedy_action_wrt_step"])
plt.xlabel("global steps")
plt.ylabel("return")
plt.show()
```


## Mahjong


To used the code for traning Mahjong agent, you will need https://github.com/pymahjong/pymahjong.


### However, we highly recommend to use the newer version of our Mahjong environment (https://github.com/Agony5757/mahjong). The newer version is much faster, and it provides more convienient APIs to the decision AI, including playing with pre-trained VLOG agents. The newer version is different from the older version in terms of some rules and fixing of some bugs.


The follows explain how to use the code for the older version (https://github.com/pymahjong/pymahjong).

### Training
To train the model, one may do
```
python run_vlog_mahjong.py --type id 1 --cql 1
```
CQL will be used if the argument "cql" is 1, and BC will be used if "cql" is 0.


### Evaluation
To train the model, one should first train the models using run_vlog_mahjong.py (We provide an example of trained model of VLOG using CQL in the supplementary material of https://openreview.net/forum?id=pjqqxepwoMy)  
To evaluate the agents, one may do
```
python eval_vlog_mahjong.py --times 8 --model_dir_0 [model_dir_0] --model_dir_1 [model_dir_1] --model_dir_2 [model_dir_2] --model_dir_3 [model_dir_3]
```
where "model_dir_i" (i=0,1,2,3) is the directory of .model file for player i on the table. 
The agents will play for 8 games (1 match) in this example. 
The result will also be saved in to a .mat file in ./data/



