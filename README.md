# Sparse Graph Tracker (SGT)
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
python projects/SGT/train_net.py --config-file projects/SGT/configs/MOT17/sgt_dla34.yaml --data-dir /root/datasets OUTPUT_DIR /root/sgt_output/mot17_val/dla34_mot17-CH
```

### Inference
```
python projects/SGT/train_net.py --config-file projects/SGT/configs/MOT17/sgt_dla34.yaml --data-dir /root/datasets --eval-only OUTPUT_DIR /root/sgt_output/mot17_test/dla34_mot17-CH
```

### Visualization
```
## GT
python projects/Datasets/MOT/vis/vis_gt.py --data-root <$DATA_ROOT> --register-data-name <e.g., mot17_train> 
python projects/Datasets/MOT/vis/vis_gt.py --data-root <$DATA_ROOT> --register-data-name <e.g., mix_crowdhuman_train> --no-video-flag 


## model output
python projects/Datasets/MOT/vis/vis_seq_from_txt_result.py --data-root <$DATA_ROOT> --result-dir <$OUTPUT_DIR> --data-name {mot17, mot20, hieve, mot17_sub, mot20_sub} --tgt-split {val,test}
```

## License
SGT is licensed under a CC-BY-NC 4.0 License.

## Citation

```BibTeX
@article{hyun2022detection,
  title={Detection Recovery in Online Multi-Object Tracking with Sparse Graph Tracker},
  author={Hyun, Jeongseok and Kang, Myunggu and Wee, Dongyoon and Yeung, Dit-Yan},
  journal={arXiv preprint arXiv:2205.00968},
  year={2022}
}
```
