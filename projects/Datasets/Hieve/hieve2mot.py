import cv2, os, shutil, argparse
import os.path as osp
import pandas as pd
import numpy as np


class Hieve2MOT():
    def __init__(self, data_root):
        data_root = osp.join(data_root, 'hieve')
        self.data_root = data_root
        self.train_root = osp.join(data_root, 'HIE20', 'train')
        self.test_root = osp.join(data_root, 'HIE20', 'test')
        self.train_video_root = osp.join(self.train_root, 'videos')
        self.test_video_root = osp.join(self.test_root, 'videos')
        self.train_label_root = osp.join(self.train_root, 'labels/track1')
        self.img_dir = 'img1' # following convention of MOT
        self.video_exts = ['.mp4', '.MOV', 'MP4']

    def organize_labels(self):
        filenames = [filename for filename in os.listdir(self.train_label_root) if '.txt' in filename ]
        for filename in filenames:
            seq_name = filename.replace('.txt', '')
            save_dir = osp.join(self.data_root, 'train', f'hieve-{seq_name}', 'gt')
            os.makedirs(save_dir,exist_ok=True)

            ## increase frame idx to start from 1 following convention of MOT
            with open(osp.join(self.train_label_root, filename), 'r') as fp:
                lines = fp.readlines()

            for idx, line in enumerate(lines):
                frame_idx = line.split(',')[0]
                lines[idx] = str(int(frame_idx)+1) + line[len(frame_idx):]

            with open(osp.join(save_dir, 'gt.txt'), 'w') as fp:
                fp.writelines(lines)

    def convert_all(self):
        self.generate_seqinfo('train')
        self.generate_seqinfo('test')
        self.extract_frames('train')
        self.extract_frames('test')
        self.organize_labels()

    def generate_seqinfo(self, mode):
        assert mode in ['train', 'test']

        video_root = self.train_video_root if mode == 'train' else self.test_video_root
        video_filenames = self.list_videos_only(video_root, self.video_exts)

        save_dir = osp.join(self.data_root, mode)
        os.makedirs(save_dir, exist_ok=True)

        for video_filename in video_filenames:
            self.generate_seqinfo_one_video(video_root, video_filename, save_dir, self.img_dir)

    def extract_frames(self, mode):
        assert mode in ['train', 'test']

        video_root = self.train_video_root if mode == 'train' else self.test_video_root
        video_filenames = self.list_videos_only(video_root, self.video_exts)

        save_root = osp.join(self.data_root, mode)
        os.makedirs(save_root, exist_ok=True)

        for video_filename in video_filenames:
            seq_name = video_filename.replace('.mp4', '').replace('.MOV', '').replace('.MP4', '')
            video_file_path = osp.join(video_root, video_filename)
            save_dir = osp.join(save_root, f"hieve-{seq_name}")
            self.extract_frames_one_video(video_file_path, save_dir, self.img_dir)

    @staticmethod
    def list_videos_only(video_root, exts):
        filenames = []
        for filename in os.listdir(video_root):
            if any([ext in filename for ext in exts]):
                filenames.append(filename)
        return filenames

    @staticmethod
    def parsing_video_meta_data(video_file_path):
        cap = cv2.VideoCapture(video_file_path)
        num_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        meta_data_dict = {}
        for d in ['num_frames', 'fps', 'width', 'height']:
            meta_data_dict[d] = eval(d)
        return meta_data_dict

    @staticmethod
    def generate_seqinfo_one_video(video_root, video_filename, save_dir, img_dir):
        video_file_path = osp.join(video_root, video_filename)
        meta_data_dict = Hieve2MOT.parsing_video_meta_data(video_file_path)

        ## save as seqinfo.ini file
        print(f"Generate seqinfo of {video_file_path}")
        seq_name = video_filename.replace('.mp4', '').replace('.MOV', '').replace('.MP4', '')
        tgt_seq_path = osp.join(save_dir, f"hieve-{seq_name}")
        os.makedirs(tgt_seq_path, exist_ok=True)
        with open(osp.join(tgt_seq_path, 'seqinfo.ini'), 'w') as fp:
            seqinfo_txt_lines = ['[Sequence]']
            seqinfo_txt_lines.append(f'name=hieve-{seq_name}')
            seqinfo_txt_lines.append(f'imDir={img_dir}')
            seqinfo_txt_lines.append(f'frameRate={meta_data_dict["fps"]}')
            seqinfo_txt_lines.append(f'seqLength={meta_data_dict["num_frames"]}')
            seqinfo_txt_lines.append(f'imWidth={meta_data_dict["width"]}')
            seqinfo_txt_lines.append(f'imHeight={meta_data_dict["height"]}')
            seqinfo_txt_lines.append(f'imExt=.jpg')
            seqinfo_txt_lines = list(map(lambda x: x + '\n', seqinfo_txt_lines))
            fp.writelines(seqinfo_txt_lines)

    @staticmethod
    def extract_frames_one_video(video_file_path, save_dir, img_dir):
        print(f"Extract frames from {video_file_path}")

        os.makedirs(save_dir, exist_ok=True)
        save_dir = osp.join(save_dir, img_dir)
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_file_path)
        success, image = cap.read()
        count = 1
        while success:
            img_path = osp.join(save_dir, f'{count:06}.jpg')
            cv2.imwrite(img_path, image)
            success, image = cap.read()
            count += 1
        cap.release()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    hieve2mot_converter = Hieve2MOT(args.data_root)
    hieve2mot_converter.convert_all()