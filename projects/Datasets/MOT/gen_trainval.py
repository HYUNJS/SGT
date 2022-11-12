import os, shutil, glob, argparse
import numpy as np
import pandas as pd
import os.path as osp


class MOT_Spliter():
    def __init__(self, data_root, tgt_data):
        assert tgt_data in ['mot17', 'mot20']
        tgt_data = tgt_data.upper()

        self.data_root = data_root
        self.tgt_data = tgt_data
        self.src_dataset_dir = osp.join(data_root, tgt_data, "train")
        self.tgt_train_dir = osp.join(data_root, f"{tgt_data}_sub", "train")
        self.tgt_val_dir = osp.join(data_root, f"{tgt_data}_sub", "val")

    def trainval_split(self):
        src_dataset_dir = self.src_dataset_dir
        tgt_train_dir = self.tgt_train_dir
        tgt_val_dir = self.tgt_val_dir

        if self.tgt_data == 'MOT17':
            seq_dirs = sorted(d for d in glob.glob(os.path.join(src_dataset_dir, "*-SDP")) if osp.isdir(d))
        elif self.tgt_data == 'MOT20':
            seq_dirs = sorted(d for d in glob.glob(os.path.join(src_dataset_dir, "*")) if osp.isdir(d))

        for seq_dir in seq_dirs:
            ## load sequence metadata
            seq_file = os.path.join(seq_dir, "seqinfo.ini")
            seq_meta = open(seq_file).read()
            seq_name = str(seq_meta[seq_meta.find('name=') + 5:seq_meta.find('\nimDir')])
            seq_length = int(seq_meta[seq_meta.find('seqLength=') + 10:seq_meta.find('\nimWidth')])

            ## split train-val
            ann_file = os.path.join(seq_dir, "gt", "gt.txt")
            seq_ann = np.loadtxt(ann_file, dtype=np.float64, delimiter=',')
            half_frame_idx = seq_length // 2
            train_flag = seq_ann[:, 0] <= half_frame_idx
            val_flag = seq_ann[:, 0] > half_frame_idx
            train_seq_ann = seq_ann[train_flag]
            val_seq_ann = seq_ann[val_flag]
            train_seq_ann = pd.DataFrame(train_seq_ann).sort_values([0, 1]).to_numpy()
            val_seq_ann = pd.DataFrame(val_seq_ann).sort_values([0, 1]).to_numpy()
            print(f"Sequence {seq_name}")
            print("# train data: {} / {}".format(train_seq_ann.shape[0], seq_ann.shape[0]))
            print("# val data: {} / {}".format(val_seq_ann.shape[0], seq_ann.shape[0]))

            ## save annotations
            train_anno_dir = osp.join(tgt_train_dir, seq_name, "gt")
            val_anno_dir = osp.join(tgt_val_dir, seq_name, "gt")
            os.makedirs(train_anno_dir, exist_ok=True)
            os.makedirs(val_anno_dir, exist_ok=True)
            np.savetxt(osp.join(train_anno_dir, "gt.txt"), train_seq_ann, fmt='%d,%d,%d,%d,%d,%d,%d,%d,%.6f')
            np.savetxt(osp.join(val_anno_dir, "gt.txt"), val_seq_ann, fmt='%d,%d,%d,%d,%d,%d,%d,%d,%.6f')

            ## copy seqinfo files
            shutil.copy(seq_file, osp.join(tgt_train_dir, seq_name, "seqinfo.ini"))
            shutil.copy(seq_file, osp.join(tgt_val_dir, seq_name, "seqinfo.ini"))

            ## make link for image folder
            src_img_dir = osp.join(seq_dir, "img1")
            tgt_train_img_dir = osp.join(tgt_train_dir, seq_name, "img1")
            tgt_val_img_dir = osp.join(tgt_val_dir, seq_name, "img1")
            if not osp.isdir(tgt_train_img_dir):
                os.symlink(src_img_dir, tgt_train_img_dir)
            if not osp.isdir(tgt_val_img_dir):
                os.symlink(src_img_dir, tgt_val_img_dir)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    mot17_spliter = MOT_Spliter(args.data_root, 'mot17')
    mot17_spliter.trainval_split()
    mot20_spliter = MOT_Spliter(args.data_root, 'mot20')
    mot20_spliter.trainval_split()

