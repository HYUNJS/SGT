import os, shutil
import os.path as osp
from commons import submit_parser, SEQ_NAMES_MOT20 as SEQ_NAMES


if __name__ == '__main__':
    args = submit_parser().parse_args()
    RESULT_DIR = osp.join(args.result_dir, 'inference', 'mot')
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    filenames = os.listdir(RESULT_DIR)
    filter_tgt_mode = lambda f: any([x in f for x in SEQ_NAMES['test']])
    filenames = list(filter(filter_tgt_mode, filenames))
    for i, filename in enumerate(filenames):
        src_filename = osp.join(RESULT_DIR, filename)
        tgt_filename = osp.join(SAVE_DIR, filename)
        shutil.copy(src_filename, tgt_filename)
        shutil.copy(src_filename, tgt_filename.replace(SEQ_NAMES['test'][i], SEQ_NAMES['val'][i]))
