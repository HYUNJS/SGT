import os, json, cv2, shutil, argparse
import os.path as osp

class CH2MIX():
    def __init__(self, data_root):
        self.data_root = data_root
        self.ch_img_train_dir = osp.join(data_root, 'crowdhuman', 'images', 'train')
        self.ch_img_val_dir = osp.join(data_root, 'crowdhuman', 'images', 'val')
        self.ch_anno_train_path = osp.join(data_root, 'crowdhuman', 'annotation_train.odgt')
        self.ch_anno_val_path = osp.join(data_root, 'crowdhuman', 'annotation_val.odgt')
        self.mix_img_dir = osp.join(data_root, 'MIX', 'crowdhuman', 'images')
        self.mix_train_dir = osp.join(self.mix_img_dir, 'train')
        self.mix_val_dir = osp.join(self.mix_img_dir, 'val')
        self.mix_list_dir = osp.join(data_root, 'MIX', 'mix_data_list')
        self.mix_dir_structure_img_train = osp.join('crowdhuman', 'images', 'train')
        self.mix_dir_structure_img_val = osp.join('crowdhuman', 'images', 'val')

    def convert_all(self):
        self.gen_label_files_train()
        self.gen_label_files_val()
        self.link_img_dataset()
        self.gen_train_val_data_list()

    def link_img_dataset(self):
        os.makedirs(self.mix_img_dir, exist_ok=True)
        if not osp.isdir(self.mix_train_dir):
            os.symlink(self.ch_img_train_dir, self.mix_train_dir)
        if not osp.isdir(self.mix_val_dir):
            os.symlink(self.ch_img_val_dir, self.mix_val_dir)

    def gen_label_files_train(self):
        label_root = osp.join(self.data_root, 'MIX', 'crowdhuman', 'labels_with_ids', 'train')
        self.gen_labels_files(self.ch_img_train_dir, label_root, self.ch_anno_train_path)

    def gen_label_files_val(self):
        label_root = osp.join(self.data_root, 'MIX', 'crowdhuman', 'labels_with_ids', 'val')
        self.gen_labels_files(self.ch_img_val_dir, label_root, self.ch_anno_val_path)

    def gen_train_val_data_list(self):
        train_img_root = osp.join(self.data_root, 'MIX', self.mix_dir_structure_img_train)
        val_img_root = osp.join(self.data_root, 'MIX', self.mix_dir_structure_img_val)
        train_data_list_filename = 'crowdhuman.train'
        val_data_list_filename = 'crowdhuman.val'
        train_img_list = [f for f in os.listdir(train_img_root) if f.endswith('jpg')]
        val_img_list = [f for f in os.listdir(val_img_root) if f.endswith('jpg')]

        self.gen_data_list(self.mix_list_dir, self.mix_dir_structure_img_train, train_data_list_filename, train_img_list)
        self.gen_data_list(self.mix_list_dir, self.mix_dir_structure_img_val, val_data_list_filename, val_img_list)

    @staticmethod
    def gen_labels_files(data_root, label_root, ann_path):
        if os.path.exists(label_root):
            shutil.rmtree(label_root)
        os.makedirs(label_root, exist_ok=True)
        anns_data = CH2MIX.load_func(ann_path)

        tid_curr = 0
        for i, ann_data in enumerate(anns_data):
            if i % 100 == 0:
                print(f"{i}/{len(anns_data)}")
            image_name = '{}.jpg'.format(ann_data['ID'])
            img_path = os.path.join(data_root, image_name)
            anns = ann_data['gtboxes']
            img = cv2.imread(
                img_path,
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            img_height, img_width = img.shape[0:2]
            for i in range(len(anns)):
                if 'extra' in anns[i] and anns[i]['extra'].get('ignore', 0) == 1:
                    continue
                x, y, w, h = anns[i]['fbox']
                v_tlx, v_tly, v_w, v_h = anns[i]['vbox']
                x += w / 2
                y += h / 2

                full_area = w * h
                visible_area = v_w * v_h
                visibility = visible_area / full_area

                label_fpath = osp.join(label_root, image_name.replace('.png', '.txt').replace('.jpg', '.txt'))
                label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    tid_curr, x / img_width, y / img_height, w / img_width, h / img_height, visibility)
                with open(label_fpath, 'a') as f:
                    f.write(label_str)
                tid_curr += 1

    @staticmethod
    def gen_data_list(mix_list_dir, seq2img_dir, data_list_filename, img_list):
        img_list.sort()
        filepath_list = [osp.join(seq2img_dir, img_name) + "\n" for img_name in img_list]

        os.makedirs(mix_list_dir, exist_ok=True)
        with open(osp.join(mix_list_dir, data_list_filename), 'w') as fp:
            for i in range(len(filepath_list)):
                fp.writelines(filepath_list[i])

    @staticmethod
    def load_func(fpath):
        print(f'Loading {fpath}')
        assert os.path.exists(fpath)
        with open(fpath, 'r') as fid:
            lines = fid.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]
        return records

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    ch2mix_converter = CH2MIX(args.data_root)
    ch2mix_converter.convert_all()