import os
import os.path as osp

DET_TYPES = ['DPM', 'FRCNN', 'SDP']

SEQ_FPS_DICT_MOT15 = {
    'ADL-Rundle-1': 30,
    'ADL-Rundle-3': 30,
    'AVG-TownCentre': 2,
    'ETH-Crossing': 14,
    'ETH-Jelmoli': 14,
    'ETH-Linthescher': 14,
    'KITTI-16': 10,
    'KITTI-19': 10,
    'PETS09-S2L2': 7,
    'TUD-Crossing': 25,
    'Venice-1': 30
}

_SEQ_FPS_DICT_MOT17 = {
    'MOT17-01': 30,
    'MOT17-02': 30,
    'MOT17-03': 30,
    'MOT17-04': 30,
    'MOT17-05': 14,
    'MOT17-06': 14,
    'MOT17-07': 30,
    'MOT17-08': 30,
    'MOT17-09': 30,
    'MOT17-10': 30,
    'MOT17-11': 30,
    'MOT17-12': 30,
    'MOT17-13': 25,
    'MOT17-14': 25,
}

SEQ_FPS_DICT_MOT20 = {
    'MOT20-01': 25,
    'MOT20-02': 25,
    'MOT20-03': 25,
    'MOT20-04': 25,
    'MOT20-05': 25,
    'MOT20-06': 25,
    'MOT20-07': 25,
    'MOT20-08': 25,
}

SEQ_FPS_DICT_MOT17 = {}
for det_type in DET_TYPES:
    SEQ_FPS_DICT_MOT17.update({f"{seq}-{det_type}": fps for seq, fps in _SEQ_FPS_DICT_MOT17.items()})

SEQ_FPS_DICT_HIEVE = {}
def fill_hieve_video_fps(data_root):
    hieve_train_root_dir = osp.join(data_root, 'hieve', 'train')
    hieve_test_root_dir = osp.join(data_root, 'hieve', 'test')

    for seq_name in os.listdir(hieve_train_root_dir):
        seq_file_path = osp.join(hieve_train_root_dir, seq_name, 'seqinfo.ini')
        seq_info = open(seq_file_path).read()
        SEQ_FPS_DICT_HIEVE[seq_name] = int(seq_info[seq_info.find('frameRate=')+10:seq_info.find('\nseqLength'):])

    for seq_name in os.listdir(hieve_test_root_dir):
        seq_file_path = osp.join(hieve_test_root_dir, seq_name, 'seqinfo.ini')
        seq_info = open(seq_file_path).read()
        SEQ_FPS_DICT_HIEVE[seq_name] = int(seq_info[seq_info.find('frameRate=')+10:seq_info.find('\nseqLength'):])

SEQ_FPS_DICT = {
    'mot15': SEQ_FPS_DICT_MOT15,
    'mot17': SEQ_FPS_DICT_MOT17,
    'mot20': SEQ_FPS_DICT_MOT20,
    'hieve': SEQ_FPS_DICT_HIEVE,
}


SEQ_NAMES_MOT15 = {
    'test': ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher',
             'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1'],
}

SEQ_NAMES_MOT17 = {
    'val': ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13'],
    'test': ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14'],
}

SEQ_NAMES_MOT20 = {
    'val': ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'],
    'test': [ 'MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08'],
}

SEQ_NAMES_HIEVE = {
    'train': [f'hieve-{i}' for i in range(1, 20)],
    'test': [f'hieve-{i}' for i in range(20, 33)],
}

SEQ_NAMES_DICT = {
    'mot15': SEQ_NAMES_MOT15,
    'mot17': SEQ_NAMES_MOT17,
    'mot20': SEQ_NAMES_MOT20,
    'hieve': SEQ_NAMES_HIEVE,
}