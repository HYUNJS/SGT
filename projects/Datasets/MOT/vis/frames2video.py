import sys
import os.path as osp
curr_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(f'{curr_dir}/../../../../')

import os, cv2, glob, argparse, time
from commons import SEQ_FPS_DICT, fill_hieve_video_fps
from multiprocessing import Pool


def convert_frame2video_mp(data_root, tgt_data_dir, tgt_seq_names, data_name):
    fill_hieve_video_fps(data_root)
    print("[READ] ", tgt_data_dir)
    tgt_data_dirs = [tgt_data_dir] * len(tgt_seq_names)
    data_types = [data_name.lower().replace('_sub', '')] * len(tgt_seq_names)
    start = time.time()
    with Pool(len(tgt_seq_names)) as p:
        p.starmap(write_seq2video, zip(tgt_data_dirs, tgt_seq_names, data_types))
    end = time.time()
    print("Takes {:.1f} sec".format(end - start))

def write_seq2video(tgt_data_dir, tgt_seq_name, data_type):
    tgt_seq_dir = osp.join(tgt_data_dir, tgt_seq_name)
    assert osp.isdir(tgt_seq_dir), f'{tgt_data_dir} is not directory or does not exist'
    fps = SEQ_FPS_DICT[data_type][tgt_seq_name]
    frames2video(tgt_seq_dir, fps)

def frames2video(frame_dir, fps):
    filenames = sorted(glob.glob(osp.join(frame_dir, '*.jpg')))
    video_filename = frame_dir + '.webm'
    print(f"Convert {frame_dir} into Video of {video_filename}")

    img_list = []
    img_size = None
    for filename in filenames:
        img = cv2.imread(filename)
        H, W, _ = img.shape
        img_size = (W, H)
        img_list.append(img)
    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'VP80'), fps, img_size)
    print("Write Video - #frames: ", len(filenames))
    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()

def main_one_video(args):
    fps = int(args.fps)
    frame_dir = args.frame_dir
    frames2video(frame_dir, fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--fps", required=True)
    args = parser.parse_args()

    frames2video(args.frame_dir, int(args.fps))
