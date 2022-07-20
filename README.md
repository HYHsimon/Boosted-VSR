# Boosted Video Super Resolution
**"Boosting Video Super Resolution with Patch-Based Temporal Redundancy Optimization"** 
by Yuhao Huang, [Hang Dong](https://sites.google.com/view/hdong/%E9%A6%96%E9%A1%B5), [Jinshan Pan](https://jspan.github.io/), Chao Zhu, Yu Guo, Ding Liu, Lean Fu, Fei Wang

## Dependencies
Same as the [BasicVSR](https://github.com/open-mmlab/mmediting).

## Test
1. Download the [Pretrained model](https://github.com/HYHsimon/Boosted-VSR/models/README.md) and [Test set](https://pan.baidu.com/s/1YvEkNOgmhQfldXzEJjcrMA?pwd=py43).

2. Run the ``test.py`` with cuda on command line: 
```bash
$python test.py --model Boosted_BasicVSR --resume /data/models/basicvsr_reds4.pth --dataset_test /data/DTVIT-test --save_path /data/hyh/DTVIT_result --gpu_ids 0
```

## Diverse Types Videos with Irregular Trajectories Dataset(DTVIT)
We have collected a new [DTVIT dataset](0https://pan.baidu.com/s/1mN21yiHykrMAWF40Vj2hqw?pwd=rkga) with more scenes which contain stationary
objects and background, including live streaming, TV program, sports live, movie and television, surveillance
camera, advertisement and first-person videos with
irregular trajectories.

The code of Boosted EDVR and more models is coming!

## Citation

If you use these models in your research, please cite:
```
@misc{boosted_vsr,
Author = {Yuhao Huang and Hang Dong and Jinshan Pan and Chao Zhu and Yu Guo and Ding Liu and Lean Fu and Fei Wang},
Title = {Boosting Video Super Resolution with Patch-Based Temporal Redundancy Optimization},
Year = {2022},
Eprint = {arXiv:2207.08674},
}
```
