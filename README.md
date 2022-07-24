# Boosted Video Super Resolution
"[**Boosting Video Super Resolution with Patch-Based Temporal Redundancy Optimization**](https://arxiv.org/abs/2207.08674)"
by Yuhao Huang, [Hang Dong](https://sites.google.com/view/hdong/%E9%A6%96%E9%A1%B5), [Jinshan Pan](https://jspan.github.io/), Chao Zhu, Yu Guo, Ding Liu, Lean Fu, Fei Wang

## Dependencies
Boosted BasicVSR and Boosted EDVR are same as the [BasicVSR](https://github.com/open-mmlab/mmediting) and [EDVR](https://github.com/xinntao/EDVR), respectively.

## Test
1. Download the [Pretrained model](https://github.com/HYHsimon/Boosted-VSR/tree/master/models) and [Test set](https://pan.baidu.com/s/1YvEkNOgmhQfldXzEJjcrMA?pwd=py43).

2. Run the ``test_basicvsr.py`` or ``test_edvr.py`` with cuda on command line: 
```bash
$python test_basicvsr.py --model Boosted_BasicVSR --resume /data/models/basicvsr_reds4.pth --dataset_test /data/DTVIT-test --save_path /data/DTVIT_result --gpu_ids 0
```
```bash
$python test_edvr.py --model Boosted_EDVR --resume_1f /data/models/edvr_1f_reds4.pth --resume_3f /data/models/edvr_3f_reds4.pth --resume_5f /data/models/edvr_5f_reds4.pth --dataset_test /data/DTVIT-test --save_path /data/DTVIT_result --gpu_ids 0
```

## Diverse Types Videos with Irregular Trajectories Dataset(DTVIT)
We have collected a new [DTVIT dataset](https://pan.baidu.com/s/1mN21yiHykrMAWF40Vj2hqw?pwd=rkga) with more scenes which contain stationary
objects and background, including live streaming, TV program, sports live, movie and television, surveillance
camera, advertisement and first-person videos with
irregular trajectories.

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
