import os
import logging
import numpy as np


MOT_FILENAME_PATTERNS = ['MOT15-', 'MOT16-', 'MOT17-', 'MOT20-']

def read_results(filename, data_type: str, is_gt=False, is_ignore=False, min_vis=0.0):
    if data_type in ('mot', 'lab'):
        read_fun = read_mot_results
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    return read_fun(filename, is_gt, is_ignore, min_vis)


def read_mot_results(filename, is_gt, is_ignore, min_vis):
    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}
    results_dict = dict()
    mot_file_flag = any([mot_filename_pattern in filename for mot_filename_pattern in MOT_FILENAME_PATTERNS])
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())

                if is_gt:
                    if mot_file_flag:
                        label = int(float(linelist[7]))
                        mark = int(float(linelist[6]))
                        vis_ratio = float(linelist[8].strip()) if len(linelist) == 9 else 1.0
                        if mark == 0 or label not in valid_labels or vis_ratio < min_vis:
                            continue
                    score = 1
                elif is_ignore:
                    if mot_file_flag:
                        label = int(float(linelist[7]))
                        if label not in ignore_labels:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    score = float(linelist[6])

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = float(linelist[1])

                results_dict[fid].append((tlwh, target_id, score))

    return results_dict


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores


def write_results(filename, results, data_type, eval_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},{conf:.3f},-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for result in results:
            frame_id = result["frame_id"]
            tlwhs = result["online_tlwhs"]
            track_ids = result["online_ids"]
            confs = result["conf"]
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, conf in zip(tlwhs, track_ids, confs):
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                if eval_type == "mot" and track_id == 0:
                    continue
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, w=w, h=h)
                f.write(line)
        f.flush()
        os.fsync(f.fileno())

    logger = logging.getLogger(__name__)
    logger.info('save results to {}'.format(filename))