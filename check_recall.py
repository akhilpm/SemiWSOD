import torch
import numpy as np
import argparse
import os
from lib.config import cfg

def main(args=None, arglist=None):
    parser = argparse.ArgumentParser(description='Check recall of box pruning', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--session', default=1, type=int, help='session to load/save model')
    parser.add_argument('-e', '--epoch', default=1, type=int, help='epoch to load model')

    cfg.ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    cfg.DATA_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, 'data'))

    if not args:
        args = parser.parse_args(arglist)

    debug_dir = os.path.join(cfg.DATA_DIR, 'debug', 'session_' + str(args.session))
    save_path = os.path.join(debug_dir, 'summary_epoch_' + str(args.epoch))
    summary = np.loadtxt(save_path)
    np.set_printoptions(suppress=True)
    print(summary[:10])
    total_gts = summary[:, 3]
    total_gts[summary[:, 3]==0] = 1.0
    ratio1 = np.divide(summary[:, 0], summary[:, 1])
    ratio2 = np.divide(summary[:, 2], summary[:, 3])
    print(np.mean(summary[:, 0]), np.mean(ratio1), np.mean(ratio2))


if __name__=="__main__":
    main()