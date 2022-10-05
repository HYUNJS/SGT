import os, argparse
import os.path as osp
import numpy as np


class MOT2MIX():
    def __init__(self, data_root, tgt_data):
        assert tgt_data in ['mot17', 'mot20']
        tgt_data = tgt_data.upper()

        self.data_root = data_root
        self.tgt_data = tgt_data
        self.mot_train_dir = osp.join(data_root, tgt_data, 'train')
        self.mot_test_dir = osp.join(data_root, tgt_data, 'test')
        self.mix_img_dir = osp.join(data_root, 'MIX', tgt_data, 'images')
        self.mix_train_dir = osp.join(self.mix_img_dir, 'train')
        self.mix_test_dir = osp.join(self.mix_img_dir, 'test')
        self.mix_list_dir = osp.join(data_root, 'MIX', 'mix_data_list')
        self.mix_dir_structure_img_train = osp.join(tgt_data, 'images', 'train')

    def convert_all(self):
        self.gen_label_files_train()
        self.link_img_dataset()
        self.gen_full_data_list()
        self.gen_train_split_data_list()

    def link_img_dataset(self):
        os.makedirs(self.mix_img_dir, exist_ok=True)
        if not osp.isdir(self.mix_train_dir):
            os.symlink(self.mot_train_dir, self.mix_train_dir)
        if not osp.isdir(self.mix_test_dir):
            os.symlink(self.mot_test_dir, self.mix_test_dir)

    def gen_label_files_train(self):
        seq_root = self.mot_train_dir
        label_root = osp.join(self.data_root, 'MIX', self.tgt_data, 'labels_with_ids', 'train')
        if self.tgt_data == 'MOT17':
            seq_names = sorted(d for d in os.listdir(seq_root) if osp.isdir(osp.join(seq_root, d)) and d.endswith("-SDP"))
        elif self.tgt_data == 'MOT20':
            seq_names = sorted(d for d in os.listdir(seq_root) if osp.isdir(osp.join(seq_root, d)))

        self.gen_label_files(seq_root, label_root, seq_names)

    def gen_full_data_list(self):
        seq_root = osp.join(self.data_root, 'MIX', self.mix_dir_structure_img_train)
        data_list_filename = f'{self.tgt_data.lower()}.train'
        full_seq_list = sorted(os.listdir(seq_root))

        self.gen_data_list(self.mix_list_dir, seq_root, self.mix_dir_structure_img_train, data_list_filename, full_seq_list)

    def gen_train_split_data_list(self):
        seq_root = osp.join(self.data_root, 'MIX', self.mix_dir_structure_img_train)
        train_data_list_filename = f'{self.tgt_data.lower()}_sub.train'
        val_data_list_filename = f'{self.tgt_data.lower()}_sub.val'
        full_seq_list = sorted(os.listdir(seq_root))

        self.gen_data_list(self.mix_list_dir, seq_root, self.mix_dir_structure_img_train, train_data_list_filename, full_seq_list, split='train')
        self.gen_data_list(self.mix_list_dir, seq_root, self.mix_dir_structure_img_train, val_data_list_filename, full_seq_list, split='val')

    @staticmethod
    def gen_label_files(seq_root, label_root, seq_names):
        tid_max = 0
        tid_offset = 0
        for seq_name in seq_names:
            seq_label_root = osp.join(label_root, seq_name, 'img1')
            os.makedirs(seq_label_root, exist_ok=True)

            seq_info = open(osp.join(seq_root, seq_name, 'seqinfo.ini')).read()
            seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

            gt_txt = osp.join(seq_root, seq_name, 'gt', 'gt.txt')
            gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
            valid_flag = np.logical_and(gt[:, 6] == 1, gt[:, 7] == 1) # only pedestrian class
            gt = gt[valid_flag]

            fids = np.unique(gt[:, 0])
            tids = np.unique(gt[:, 1])
            tid_dict = {int(t):idx+1 for idx, t in enumerate(tids)}
            for fid in fids:
                curr_gt = gt[gt[:, 0] == fid]
                if len(curr_gt) == 0:
                    continue
                sort_idx = np.argsort(curr_gt[:, 1])
                curr_gt = curr_gt[sort_idx]

                label_lines = []
                label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(int(fid)))

                ## annotation format: frm_idx, obj_id, x, y, w, h, confidence, category, visibility
                for _, tid, x, y, w, h, _, _, vis in curr_gt:
                    updated_tid = tid_dict[int(tid)] + tid_offset
                    x += w / 2
                    y += h / 2
                    label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        updated_tid, x / seq_width, y / seq_height, w / seq_width, h / seq_height, vis)
                    label_lines.append(label_str)

                with open(label_fpath, 'w') as f:
                    f.writelines(label_lines)
                    tid_max = max(tid_max, updated_tid)

            tid_offset += tids.shape[0]

    @staticmethod
    def gen_data_list(mix_list_dir, seq_root, seq2img_dir, data_list_filename, seq_list, split=None):
        filepath_list = []
        for seq_name in seq_list:
            if not osp.isdir(osp.join(seq_root, seq_name)):
                continue

            img_root = osp.join(seq_root, seq_name, 'img1')
            filename_list = os.listdir(img_root)
            filename_list.sort()
            half_frame_idx = len(filename_list) // 2
            if split == 'train':
                filename_list = filename_list[:half_frame_idx]
            elif split == 'val':
                filename_list = filename_list[half_frame_idx:]
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
    mot17_2mix_converter = MOT2MIX(args.data_root, 'mot17')
    mot17_2mix_converter.convert_all()
    mot20_2mix_converter = MOT2MIX(args.data_root, 'mot20')
    mot20_2mix_converter.convert_all()

