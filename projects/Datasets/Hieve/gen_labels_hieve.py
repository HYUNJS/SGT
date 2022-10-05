import os, argparse
import os.path as osp
import numpy as np


class Hieve2MIX():
    def __init__(self, data_root):
        self.data_root = data_root
        self.hieve_train_dir = osp.join(data_root, 'hieve', 'train')
        self.hieve_test_dir = osp.join(data_root, 'hieve', 'test')
        self.mix_img_dir = osp.join(data_root, 'MIX', 'hieve', 'images')
        self.mix_train_dir = osp.join(self.mix_img_dir, 'train')
        self.mix_test_dir = osp.join(self.mix_img_dir, 'test')
        self.mix_list_dir = osp.join(data_root, 'MIX', 'mix_data_list')
        self.mix_dir_structure_img_train = osp.join('hieve', 'images', 'train')

    def convert_all(self):
        self.gen_label_files_train()
        self.link_img_dataset()
        self.gen_full_data_list()
        self.gen_train_split_data_list()

    def link_img_dataset(self):
        os.makedirs(self.mix_img_dir, exist_ok=True)
        if not osp.isdir(self.mix_train_dir):
            os.symlink(self.hieve_train_dir, self.mix_train_dir)
        if not osp.isdir(self.mix_test_dir):
            os.symlink(self.hieve_test_dir, self.mix_test_dir)

    def gen_label_files_train(self):
        seq_root = self.hieve_train_dir
        label_root = osp.join(self.data_root, 'MIX', 'hieve', 'labels_with_ids', 'train')
        self.gen_label_files(seq_root, label_root)

    def gen_full_data_list(self):
        seq_root = osp.join(self.data_root, 'MIX', self.mix_dir_structure_img_train)
        data_list_filename = 'hieve.train'
        full_seq_list = os.listdir(seq_root)

        self.gen_data_list(self.mix_list_dir, seq_root, self.mix_dir_structure_img_train, data_list_filename, full_seq_list)

    def gen_train_split_data_list(self):
        seq_root = osp.join(self.data_root, 'MIX', self.mix_dir_structure_img_train)
        train_data_list_filename = 'hieve_sub.train'
        val_data_list_filename = 'hieve_sub.val'
        train_seq_list = [f'hieve-{i}' for i in range(1, 16)]
        val_seq_list = [f'hieve-{i}' for i in range(16, 20)]

        self.gen_data_list(self.mix_list_dir, seq_root, self.mix_dir_structure_img_train, train_data_list_filename, train_seq_list)
        self.gen_data_list(self.mix_list_dir, seq_root, self.mix_dir_structure_img_train, val_data_list_filename, val_seq_list)

    @staticmethod
    def gen_label_files(seq_root, label_root):
        seqs = [dirname for dirname in os.listdir(seq_root) if osp.isdir(osp.join(seq_root, dirname))]

        tid_offset = 0
        for seq_name in seqs:
            seq_label_root = osp.join(label_root, seq_name, 'img1')
            os.makedirs(seq_label_root, exist_ok=True)

            seq_info = open(osp.join(seq_root, seq_name, 'seqinfo.ini')).read()
            seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

            gt_txt = osp.join(seq_root, seq_name, 'gt', 'gt.txt')
            gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
            fids = np.unique(gt[:, 0])
            tids = np.unique(gt[:, 1])

            for fid in fids:
                curr_gt = gt[gt[:, 0] == fid]
                if len(curr_gt) == 0:
                    continue
                sort_idx = np.argsort(curr_gt[:, 1])
                curr_gt = curr_gt[sort_idx]

                label_lines = []
                label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(int(fid)))
                for _, tid, x, y, w, h, mark, _, _, _ in curr_gt:
                    if mark == 0:
                        continue
                    updated_tid = int(tid + tid_offset)

                    x += w / 2
                    y += h / 2
                    label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(updated_tid, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
                    label_lines.append(label_str)

                with open(label_fpath, 'w') as f:
                    f.writelines(label_lines)

            tid_offset += int(tids.max())

    @staticmethod
    def gen_data_list(mix_list_dir, seq_root, seq2img_dir, data_list_filename, seq_list):
        filepath_list = []
        for seq_name in seq_list:
            if not osp.isdir(osp.join(seq_root, seq_name)):
                continue

            img_root = osp.join(seq_root, seq_name, 'img1')
            filename_list = os.listdir(img_root)
            filename_list.sort()
            filepath_list.append([osp.join(seq2img_dir, seq_name, 'img1', filename) + "\n" for filename in filename_list])

        os.makedirs(mix_list_dir, exist_ok=True)
        with open(osp.join(mix_list_dir, data_list_filename), 'w') as fp:
            for i in range(len(filepath_list)):
                fp.writelines(filepath_list[i])

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    mot2mix_converter = Hieve2MIX(args.data_root)
    mot2mix_converter.convert_all()

