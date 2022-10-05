import os, argparse, shutil
import os.path as osp
import numpy as np
import pandas as pd


SEQ_NAMES = {
    'test': [f'hieve-{i}' for i in range(20, 33)],
}


def adjust_format_v1(filename):
    df = pd.read_csv(osp.join(DATA_ROOT, filename), header=None)
    ## hieve starts frame index from 0 rather than 1
    df[0] = df[0] - 1

    ## hieve bbox in integer
    df[2] = df[2].round().astype(int)
    df[3] = df[3].round().astype(int)
    df[4] = df[4].round().astype(int)
    df[5] = df[5].round().astype(int)

    ## adjust x, y coordinates as them starting from 1
    x_negative_mask = df[2] < 1
    y_negative_mask = df[3] < 1
    x_offset = 1 - df.loc[x_negative_mask, 2]
    y_offset = 1 - df.loc[y_negative_mask, 3]

    df.loc[x_negative_mask, 2] += x_offset
    df.loc[y_negative_mask, 3] += y_offset
    df.loc[x_negative_mask, 4] -= x_offset
    df.loc[y_negative_mask, 5] -= y_offset

    ws = df.loc[:, 4].to_numpy()
    hs = df.loc[:, 5].to_numpy()
    ws.clip(min=1, out=ws)
    hs.clip(min=1, out=hs)
    df.loc[:, 4] = ws
    df.loc[:, 5] = hs

    df.to_csv(osp.join(SAVE_ROOT, filename.replace('hieve-', '')), index=False, header=None, float_format='%.3f')

def adjust_format_v2(filename):
    df = pd.read_csv(osp.join(DATA_ROOT, filename), header=None)
    ## hieve starts frame index from 0 rather than 1
    df[0] = df[0] - 1

    df.to_csv(osp.join(SAVE_ROOT, filename.replace('hieve-', '')), index=False, header=None, float_format='%.3f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--prj", required=True)
    parser.add_argument("--result-dir")
    args = parser.parse_args()

    DATA_ROOT = osp.join(args.result_dir, 'inference/mot')
    SAVE_ROOT = osp.join(args.result_dir, 'submit')
    assert osp.isdir(DATA_ROOT), f'Not exist {DATA_ROOT}'

    filenames = os.listdir(DATA_ROOT)
    filter_test_result = lambda f: any([x in f for x in SEQ_NAMES['test']])
    filenames = list(filter(filter_test_result, filenames))
    assert len(filenames) > 0, "No test sequence output files"

    os.makedirs(SAVE_ROOT, exist_ok=True)

    for filename in filenames:
        adjust_format_v1(filename)
        # adjust_format_v2(filename)