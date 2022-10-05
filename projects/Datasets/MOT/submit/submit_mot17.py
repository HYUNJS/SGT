import os, shutil
import os.path as osp
from commons import submit_parser, DET_TYPES, SEQ_NAMES_MOT17 as SEQ_NAMES


if __name__ == '__main__':
    args = submit_parser().parse_args()
    RESULT_DIR = osp.join(args.result_dir, 'inference', 'mot')
    SAVE_DIR = osp.join(args.result_dir, 'submission')
    # SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    filenames = os.listdir(RESULT_DIR)
    filter_tgt_mode = lambda f: any([x in f for x in SEQ_NAMES['test']])
    filenames = list(filter(filter_tgt_mode, filenames))

    det_type_idx = 0
    for i, det_type in enumerate(DET_TYPES):
        if det_type in filenames[0]:
            det_type_idx = i

    for det_type in DET_TYPES:
        for i, filename in enumerate(filenames):
            src_filename = osp.join(RESULT_DIR, filename)
            tgt_filename = osp.join(SAVE_DIR, filename.replace(DET_TYPES[det_type_idx], det_type))
            shutil.copy(src_filename, tgt_filename)
            shutil.copy(src_filename, tgt_filename.replace(SEQ_NAMES['test'][i], SEQ_NAMES['val'][i]))
