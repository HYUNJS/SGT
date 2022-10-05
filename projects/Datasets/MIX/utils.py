import os
import os.path as osp
import numpy as np
import pandas as pd


def split_mot_datalist_into_half(datalist_file_dir, tgt_filename, save_train_name, save_val_name):
    assert 'mot' in tgt_filename

    ## read original filelist
    datalist_file_path = osp.join(datalist_file_dir, tgt_filename)
    with open(datalist_file_path, 'r') as fp:
        img_file_list = fp.readlines()

    ## convert img filename list into dictionary which key is sequence name
    # img_file_arr = np.char.strip(np.array(img_file_list))
    img_file_arr = np.array(img_file_list) ## do not remove newline flag so that it is preserved when we make new file
    seq_names = pd.Series(img_file_arr).str.split('/').str[3].values
    unique_seq_names = np.unique(seq_names)
    img_file_dict = {}
    for tgt_seq_name in unique_seq_names:
        img_file_dict[tgt_seq_name] = img_file_arr[seq_names == tgt_seq_name]

    ## split into train_sub and val_sub
    img_file_train_sub_dict, img_file_val_sub_dict = {}, {}
    for tgt_seq_name in unique_seq_names:
        half_idx = len(img_file_dict[tgt_seq_name]) // 2
        img_file_train_sub_dict[tgt_seq_name] = img_file_dict[tgt_seq_name][:half_idx]
        img_file_val_sub_dict[tgt_seq_name] = img_file_dict[tgt_seq_name][half_idx:]

    ## output into file
    with open(osp.join(datalist_file_dir, save_train_name), 'w') as fp:
        for tgt_seq_name in unique_seq_names:
            fp.writelines(img_file_train_sub_dict[tgt_seq_name])

    with open(osp.join(datalist_file_dir, save_val_name), 'w') as fp:
        for tgt_seq_name in unique_seq_names:
            fp.writelines(img_file_val_sub_dict[tgt_seq_name])

def gen_labels_mot(seq_root, label_root):
    os.makedirs(label_root, exist_ok=True)
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1
    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

        seq_label_root = osp.join(label_root, seq, 'img1')
        os.makedirs(seq_label_root, exist_ok=True)

        for fid, tid, x, y, w, h, mark, label, vis in gt:
            if mark == 0 or label != 1:
                continue
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height, vis)
            with open(label_fpath, 'a') as f:
                f.write(label_str)

if __name__ == '__main__':
    pass
    ## how to use split_mot_datalist_into_half
    # datalist_file_dir = '/mnt/video_nfs4/video_recognition/detectron2_datasets/MIX/mix_data_list'
    # tgt_filename = 'mot20.train'
    # save_train_name  = 'mot20sub.train'
    # save_val_name = 'mot20sub.val'
    # split_mot_datalist_into_half(datalist_file_dir, tgt_filename, save_train_name, save_val_name)

    ## how to use gen_labels_mot
    # seq_root = '/mnt/video_nfs4/video_recognition/detectron2_datasets/MIX/MOT17/images/train'
    # # label_root = '/mnt/video_nfs4/video_recognition/detectron2_datasets/MIX/MOT17/new_labels_with_ids/train'
    # label_root = '/mnt/video_nfs4/video_recognition/detectron2_datasets/MIX/MOT17/labels_with_ids/train'
    # gen_labels_mot(seq_root, label_root)


    seq_root = '/mnt/video_nfs4/video_recognition/detectron2_datasets/MIX/MOT17/images/train'
    # # label_root = '/mnt/video_nfs4/video_recognition/detectron2_datasets/MIX/MOT17/new_labels_with_ids/train'
    # label_root = '/mnt/video_nfs4/video_recognition/detectron2_datasets/MIX/MOT17/labels_with_ids/train'
    # gen_labels_mot(seq_root, label_root)
