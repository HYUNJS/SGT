import sys
import os.path as osp
curr_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(f'{curr_dir}/../../../../')

import os, argparse, time
from PIL import Image
import multiprocessing as mp

from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.config import get_cfg

from projects.Datasets.MOT.build import get_mot_dataset_dicts
from projects.Datasets.MIX.build import get_mix_dataset_dicts
from projects.Datasets.MOT.builtin import register_all_mot, SPLITS as MOT_SPLITS
from projects.Datasets.MIX.builtin import register_mix_tgt, SPLITS as MIX_SPLITS
from projects.Datasets.MOT.config import add_mot_dataset_config
from projects.Datasets.MIX.config import add_mix_dataset_config

from simple_visualizer import SimpleVisualizer, imwrite
from frames2video import convert_frame2video_mp


def vis_dataset(dataset_dict, metadata, save_dir):
    for d in dataset_dict:
        img_name = d['file_name']
        img = convert_PIL_to_numpy(Image.open(img_name), 'RGB')
        visualizer = SimpleVisualizer(img, metadata)
        vis_output = visualizer.draw_dataset_dict_simple(d)
        save_path = osp.join(save_dir, d['sequence_name'])
        os.makedirs(save_path, exist_ok=True)
        imwrite(osp.join(save_path, img_name.split('/')[-1]), vis_output.get_image())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument("--register-data-name", required=True, help='e.g. mix_mot2ub_train')
    parser.add_argument("--min-visibility", default=0.0)
    parser.add_argument("--no-video-flag", action="store_true",)
    args = parser.parse_args()

    data_root = args.data_root
    register_data_name = args.register_data_name
    min_visibility = args.min_visibility

    meta = MetadataCatalog.get(register_data_name)
    cfg = get_cfg()
    add_mot_dataset_config(cfg)
    add_mix_dataset_config(cfg)
    cfg.TEST.EVAL_MIN_VISIBILITY = min_visibility
    cfg.INPUT.TRAIN_MIN_VISIBILITY = min_visibility
    cfg.DATASETS.TRAIN = (register_data_name, )
    register_mix_tgt(data_root, cfg)
    register_all_mot(data_root, cfg)

    mix_dataset_flag = 'mix_' in register_data_name
    SPLITS = MIX_SPLITS if mix_dataset_flag else MOT_SPLITS
    dataset_name, split = [s for s in SPLITS if s[0] == register_data_name][0][1:3]
    save_dir = [data_root, dataset_name.replace('mot', 'MOT'), 'gt_vis', split]
    num_core = min(mp.cpu_count(), 16)

    ## load dataset
    print('Loading {}'.format(register_data_name))
    if 'mix_' in register_data_name:
        save_dir.insert(1, 'MIX')
        save_dir = osp.join(*save_dir)
        dataset_dicts, _ = get_mix_dataset_dicts([register_data_name], filter_empty=True, filter_pairless=False)
        dataset_dicts = dataset_dicts[0]
    else:
        save_dir = osp.join(*save_dir)
        dataset_dicts = get_mot_dataset_dicts([register_data_name], filter_empty=True)

    ## visualize gt frames
    size_per_core = len(dataset_dicts) // num_core + 1
    splited_dataset_dicts = [dataset_dicts[i*size_per_core:(i+1)*size_per_core] for i in range(num_core)]
    print(f"Visualizing {register_data_name} - {len(dataset_dicts)} frames")
    start = time.time()
    with mp.Pool(num_core) as p:
        p.starmap(vis_dataset, zip(splited_dataset_dicts, [meta] * num_core, [save_dir] * num_core))
    end = time.time()
    print("Takes {:.1f} sec".format(end-start))

    ## convert into video
    if not args.no_video_flag:
        convert_frame2video_mp(data_root, save_dir, os.listdir(save_dir), dataset_name)