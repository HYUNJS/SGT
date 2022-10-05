import argparse


def add_dataset_parser(detectron2_parser):
    parser = argparse.ArgumentParser(parents=[detectron2_parser], add_help=False)
    parser.add_argument("--data-dir", default="datasets", type=str)
    return parser