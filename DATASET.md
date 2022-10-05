
## Preparing Dataset
 * After following the steps below, the datasets will be organized as follows:
```
$DATA_ROOT/
  MOT17/
    train/
    test/
  MOT17_sub/
    train/
    val/
  MOT20/
  ...
  MIX/ # for jointly training different datasets
    mix_data_list/
      mot17_sub.train
      mot17_sub.val
      crowdhuman.train
      ...
    MOT17/
      images/
        train/
        test/
      labels_with_ids/
        train/
    crowdhuman/
    ...
```


### MOT17/20 Dataset
1) Download from https://motchallenge.net/
   * Due to slow download speed, we share the datasets via OneDrive [MOT17](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jhyunaa_connect_ust_hk/EdE97TxI-kpIqdznWT83LhsB3yUjLoLiMskhGI20-eZYlw?e=ifdbEg) and [MOT20](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jhyunaa_connect_ust_hk/EUjH1AsnvYtLjtBGPQbFGawBEgDJB8-cROCkbQJDTdZLWw?e=izHudy)
2) Organize as below:
```
$DATA_ROOT/
  MOT17/
    train/
      MOT17-02-SDP/
        seqinfo.ini
        gt/
          gt.txt
        img1/
          000001.jpg
          ...
    test/
      MOT17-01-SDP/
      ...
```
3) Split train-val set - Use the first half frames as train set
```
python projects/Datasets/MOT/gen_trainval.py --data-root <$DATA_ROOT>
```
4) Covnert MOT format to MIX format
```
python projects/Datasets/MOT/gen_labels_mot.py --data-root <$DATA_ROOT>
```

### CrowdHuman dataset
1) Download from https://www.crowdhuman.org/
2) Organize as below:
```
$DATA_ROOT/
  crowdhuman/
    annotation_train.odgt
    annotation_val.odgt
    images/
      train/
        *.jpg
      val/
        *.jpg
```
3) Convert given odgt file into MIX format
```
python projects/Datasets/CrowdHuman/gen_labels_crowdhuman.py --data-root <$DATA_ROOT>
```


### HiEve Dataset
1) Download from http://humaninevents.org/newdownload.html
2) Organize as below:
```
$DATA_ROOT/
  hieve/
    HIE20/
      train/
        labels/
          track1/
            *.txt
        videos/
          *.MP4
          *.MOV
          *.MP4
      test/
        videos/
          *.mp4
```
3) Convert hieve dataset into MOT dataset format by running the script below:
```
python projects/Datasets/Hieve/hieve2mot.py --data-root <$DATA_ROOT>
```
4) Convert MOT format to MIX format
```
python projects/Datasets/Hieve/gen_labels_hieve.py --data-root <$DATA_ROOT>
```
