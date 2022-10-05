import sys
import os.path as osp
curr_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(f'{curr_dir}/../../../../')

import os, torch, time, argparse
from projects.Datasets.MOT.evaluation.io import read_results
from projects.Datasets.MOT.vis.simple_visualizer import SimpleVisualizer, imwrite
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances
from PIL import Image, ImageOps
from fvcore.common.file_io import PathManager
from detectron2.data.detection_utils import convert_PIL_to_numpy
from multiprocessing import Pool

from frames2video import convert_frame2video_mp
from commons import SEQ_NAMES_DICT


def read_image(file_name, format='RGB'):
     with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)
        image = ImageOps.exif_transpose(image)
        return convert_PIL_to_numpy(image, format)

def save_vis_img(frame_id, tgt_dict, seq_name, vis_conf=False, min_conf=0.0):
    bbox_per_image, tid_per_image, score_per_image = [], [], []
    for row in tgt_dict[frame_id]:
        bbox, tid, score = torch.tensor(row[0]), torch.tensor(int(row[1])), torch.tensor(row[2])
        if score < min_conf:
            continue
        bbox[0:2] = bbox[0:2] - 1
        bbox[2:] = bbox[2:] + bbox[0:2]
        bbox_per_image.append(bbox)
        tid_per_image.append(tid)
        score_per_image.append(score)
    bbox_per_image = torch.stack(bbox_per_image)
    box_ids_per_image = torch.stack(tid_per_image)
    score_per_image = torch.stack(score_per_image)
    # label_per_image = torch.tensor([0] * score_per_image.size(0))

    img_path = os.path.join(IMG_DIR, seq_name, 'img1/{:06}.jpg'.format(frame_id))
    img = read_image(img_path)
    image_size = img.shape[0:2]
    result = Instances(image_size)
    result.pred_boxes = Boxes(bbox_per_image)
    result.scores = score_per_image
    # result.pred_classes = label_per_image
    result.pred_ids = box_ids_per_image

    metadata = MetadataCatalog.get("__unused")
    sub_metadata = MetadataCatalog.get("__unused")
    instance_mode = ColorMode.IMAGE
    visualizer = SimpleVisualizer(img, metadata, sub_metadata, instance_mode=instance_mode, vis_conf=vis_conf, tgt_ignore_id=-1)
    vis_output = visualizer.draw_instance_predictions(predictions=result)
    save_path = os.path.join(SAVE_ROOT, seq_name, '{:06}.jpg'.format(frame_id))
    imwrite(save_path, vis_output.get_image())

def process_seq(filename):
    seq_name = filename.split('.')[0]
    print("VIS ", seq_name)
    save_dir = os.path.join(SAVE_ROOT, seq_name)
    os.makedirs(save_dir, exist_ok=True)
    tgt_filename = os.path.join(TXT_DIR, filename)
    try:
        tgt_dict = read_results(tgt_filename, 'mot', is_gt=False)
        frame_ids = list(tgt_dict.keys())
        for frame_id in frame_ids:
            save_vis_img(frame_id, tgt_dict, seq_name, score_flag, min_score)
    except Exception as e:
        print(e.message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Absolute directory of dataset - e.g., /root/datasets/")
    parser.add_argument("--result-dir", required=True, help="Absolute directory of output - e.g., /root/sgt_output/dla34_mot17-CH")
    parser.add_argument("--data-name", required=True, choices=['mot17', 'mot20', 'hieve', 'mot17_sub', 'mot17_sub', 'hieve_sub'])
    parser.add_argument("--tgt-split", required=True, choices=['val', 'test'])
    parser.add_argument("--no-video-flag", action="store_true")
    parser.add_argument("--score-flag", action="store_true")
    parser.add_argument("--det-flag", action='store_true')
    parser.add_argument("--min-score", default=0.0)
    args = parser.parse_args()

    ## parsing necessary info
    score_flag = args.score_flag
    min_score = float(args.min_score)
    IMG_DIR = osp.join(args.data_root, args.data.replace('mot', 'MOT'), args.tgt_split) # '${IMG_DIR}/{seq_name}/img1/{:06}.jpg'.format(frame_id)
    TXT_DIR = osp.join(args.result_dir, 'inference', 'det' if args.det_flag else 'mot')
    SAVE_ROOT = osp.join(args.result_dir, 'visualization')
    SEQ_NAMES = SEQ_NAMES_DICT[args.data_name][args.tgt_split]
    print("[READ] ", TXT_DIR)
    print("[SAVE] ", SAVE_ROOT)

    ## get target sequence names
    filenames = os.listdir(TXT_DIR)
    filenames = [f for f in filenames if '.txt' in f]
    filter_tgt_split = lambda f: any([x in f for x in SEQ_NAMES])
    filenames = list(filter(filter_tgt_split, filenames))

    ## visualize each sequence - all frames
    start = time.time()
    with Pool(len(filenames)) as p:
        p.map(process_seq, filenames)
    end = time.time()
    print("Takes {:.1f} sec".format(end-start))

    ## convert into video
    if not args.no_video_flag:
        tgt_seq_names = [seq_name for seq_name in os.listdir(SAVE_ROOT) if osp.isdir(osp.join(SAVE_ROOT, seq_name))]
        tgt_seq_names = list(filter(filter_tgt_split, tgt_seq_names))
        convert_frame2video_mp(args.data_root, SAVE_ROOT, tgt_seq_names, args.data_name)