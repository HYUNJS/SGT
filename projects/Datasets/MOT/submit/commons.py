import argparse


SEQ_NAMES_MOT16 = {
    'val': ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13'],
    'test': ['MOT16-01', 'MOT16-03', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14'],
}

SEQ_NAMES_MOT17 = {
    'val': ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13'],
    'test': ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14'],
}

SEQ_NAMES_MOT20 = {
    'val': ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'],
    'test': ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08'],
}

DET_TYPES = ['DPM', 'FRCNN', 'SDP']

def submit_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True) # ${RESULT_DIR}/inference/mot
    parser.add_argument("--save-dir", default='')
    return parser