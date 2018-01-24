# PySC2 bot in Pytorch

## Reference:
Relies on the repo https://github.com/simonmeister/pysc2-rl-agents for implementations 
of action / observation space pre-processing, network architectures. 

Entire backend is in Pytorch, with TF for tensorboard.

## To run training:
```bash
python run.py --envs 32 --map MoveToBeacon
```


## Result:
**MoveToBeacon**

MoveToBeacon           |  CollectMineralShards
:-------------------------:|:-------------------------:
![](imgs/result_MoveToBeacon.png)  |  ![](imgs/result_CollectMineralShards.png)


## TODO:
- [ ] Train on other mini-games
- [ ] Use replay data
- [ ] Optimize the Runner to work with Torch tensor instead of numpy array
- [ ] Multi-GPU training