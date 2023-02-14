# Sparse Graph Tracker (SGT)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/detection-recovery-in-online-multi-object/multi-object-tracking-on-hieve)](https://paperswithcode.com/sota/multi-object-tracking-on-hieve?p=detection-recovery-in-online-multi-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/detection-recovery-in-online-multi-object/multi-object-tracking-on-mot16)](https://paperswithcode.com/sota/multi-object-tracking-on-mot16?p=detection-recovery-in-online-multi-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/detection-recovery-in-online-multi-object/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=detection-recovery-in-online-multi-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/detection-recovery-in-online-multi-object/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=detection-recovery-in-online-multi-object)

Official code for Sparse Graph Tracker (SGT) based on the Detectron2 framework. Please feel free to leave an ISSUE or send me an email (jhyunaa@ust.hk).

## News
* (2022.10.11) Our paper is accepted WACV 2023! (arxiv paper will be updated soon)
* (2022.10.06) Code and pretrained weights are released!

## Installation
* Please refer [INSTALL.md](INSTALL.md) for the details 
## Dataset Setup
* Please refer [DATASET.md](DATASET.md) for the details

## Model Zoo
- [download model weights](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jhyunaa_connect_ust_hk/ErLZ6DG6CndHs-Lo12AxZKAB1kb4AJCh8adtnRAlXTNuzA?e=rKeboA)
* Please modify the path of checkpoints in the config file based on your checkpoint directory

### MOT17
| Name | Dataset | HOTA | MOTA | IDF1| Download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| SGT | MOT17 | 58.2 | 73.2 | 70.2 | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jhyunaa_connect_ust_hk/EQW4mXblacdDtVc3uxaTsXYB2yqqUTQv9cwnBipAnpKblA?e=9pT2RO) |
| SGT | MOT17 + CrowdHuman | 60.8 | 76.4 | 72.8 | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jhyunaa_connect_ust_hk/EST0ZaRgqvlJoW2TGFpCUToBRUzGRkgXZQva32rypzWdZQ?e=OzCVgI) |

### MOT20
| Name | Dataset | HOTA | MOTA | IDF1| Download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| SGT | MOT20 | 51.6 | 64.5 | 62.7 | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jhyunaa_connect_ust_hk/EVQks100QaRNp81QlSoQbwMBgncyxw-4cmE_eIrR3JPJoA?e=hQIHxF) |
| SGT | MOT20 + CrowdHuman | 57.0 | 72.8 | 70.6 | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jhyunaa_connect_ust_hk/EZWXTwJFbPNBs2d32RVAa84BYTjImlYMSsz-Fp4lt8aE6A?e=qiOoHw) |

### HiEve
| Name | Dataset | MOTA | IDF1 | Download |
| :---: | :---: | :---: | :---: | :---: | 
| SGT | HiEve | 47.2 | 53.7 | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jhyunaa_connect_ust_hk/EVQks100QaRNp81QlSoQbwMBgncyxw-4cmE_eIrR3JPJoA?e=hQIHxF) |

## How to run?

### Train
```
python projects/SGT/train_net.py --config-file projects/SGT/configs/MOT17/sgt_dla34.yaml --data-dir /root/datasets --num-gpus 2 OUTPUT_DIR /root/sgt_output/mot17_val/dla34_mot17-CH
```

### Inference
```
python projects/SGT/train_net.py --config-file projects/SGT/configs/MOT17/sgt_dla34.yaml --data-dir /root/datasets --num-gpus 1 --eval-only OUTPUT_DIR /root/sgt_output/mot17_test/dla34_mot17-CH
```

### Visualization
```
## GT
python projects/Datasets/MOT/vis/vis_gt.py --data-root <$DATA_ROOT> --register-data-name <e.g., mot17_train> 
python projects/Datasets/MOT/vis/vis_gt.py --data-root <$DATA_ROOT> --register-data-name <e.g., mix_crowdhuman_train> --no-video-flag 


## model output
python projects/Datasets/MOT/vis/vis_seq_from_txt_result.py --data-root <$DATA_ROOT> --result-dir <$OUTPUT_DIR> --data-name {mot17, mot20, hieve, mot17_sub, mot20_sub} --tgt-split {val,test}
```

## Motivation
![image](https://user-images.githubusercontent.com/29353227/194476858-69c24328-f461-48b9-9262-17f90f38e652.png)

## Pipeline
![image](https://user-images.githubusercontent.com/29353227/194477178-d31da80b-c215-4acf-ab9d-8519b9f54f9f.png)

## MOT Benchmark Results
![image](https://user-images.githubusercontent.com/29353227/194478496-39309fea-ced0-4d3f-8be0-cce87f4c9c57.png)

## Ablation Experiment Results
![image](https://user-images.githubusercontent.com/29353227/194478002-ba6bff6d-6665-45de-80ed-51f384b10094.png)

![image](https://user-images.githubusercontent.com/29353227/194478011-66d3c56d-89bc-40e5-a4d0-22558a9d9159.png)

## Visualization
![image](https://user-images.githubusercontent.com/29353227/194478129-4a1684ee-7326-4ad1-b3d5-989b13e2b7c5.png)



## License
Code of SGT is licensed under the CC-BY-NC 4.0 license and free for research and academic purpose.
SGT is based on the framework [Detectron2](https://github.com/facebookresearch/detectron2) which is released under the Apache 2.0 license and the detector [CenterNet](https://github.com/xingyizhou/CenterNet) which is released under the MIT license.
This codebase also provides Detectron2 version of [FairMOT](https://github.com/ifzhang/FairMOT) which is released under the MIT license.

## Citation

@inproceedings{hyun2023detection,
  title={Detection recovery in online multi-object tracking with sparse graph tracker},
  author={Hyun, Jeongseok and Kang, Myunggu and Wee, Dongyoon and Yeung, Dit-Yan},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4850--4859},
  year={2023}
}
