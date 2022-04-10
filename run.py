import lib  # add lib folder to sys.path
import os
import sys
import logging
import time
import argparse
import torch
from colorama import init, deinit, Back, Fore
from config import cfg
lib_path = os.path.join(os.path.dirname("lib/"))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
from script.train import train
from script.test import test
from script.detect import detect
from script.hyp_test import hyp_test
from script.cam_train import cam_train
from script.cam_test import cam_test
from utils.net_utils import parse_additional_params


def add_common_parser_arguments(parser):
    parser.add_argument('-n', '--net', default='vgg16',
                        help='backbone for faster rcnn network',
                        choices=['vgg16', 'resnet18', 'resnet34', 'resnet50',
                                 'resnet101', 'resnet152'])
    parser.add_argument('-d', '--dataset', default='voc_2007_trainval',
                        help='dataset (classes) to load')
    parser.add_argument('-s', '--session', default=1, type=int,
                        help='session to load/save model')
    parser.add_argument('-e', '--epoch', default=1, type=int,
                        help='epoch to load model')
    parser.add_argument('-cag', '--class_agnostic', action='store_true',
                        help='whether perform class agnostic bounding box regression')
    parser.add_argument('--cuda', action='store_true',
                        help='whether use CUDA for network')
    parser.add_argument('--mGPU', action='store_true',
                        help='whether use multi GPU for network (train only)')
    parser.add_argument("-vp", "--visdom_port", type=int, help="visdom port to run", default=9990)


parser = argparse.ArgumentParser(description='Faster R-CNN Network')
subparsers = parser.add_subparsers(dest='mode', help='main mode of network')
formatter = argparse.ArgumentDefaultsHelpFormatter

# create the parser for the train mode
parser_train = subparsers.add_parser('train', formatter_class=formatter,
                                     help='help for TRAIN mode of network')
add_common_parser_arguments(parser_train)
parser_train.add_argument('-bs', '--batch_size', type=int, default=None,
                          help='training batch size')
parser_train.add_argument('-lr', '--learning_rate', type=float, default=None,
                          help='training learning rate')
parser_train.add_argument('-o', '--optimizer', choices=['sgd', 'adam'],
                          default='sgd', help='training optimizer')
parser_train.add_argument('-lrds', '--lr_decay_step', type=int, default=None,
                          help='learning rate decay step, in epochs')
parser_train.add_argument('-lrdg', '--lr_decay_gamma', type=float, default=None,
                          help='learning rate decay ratio')
parser_train.add_argument('-p', '--pretrain', action='store_true',
                          help='load weigths from checkpoint or not '
                          + 'Need to set SESSION and EPOCH')
parser_train.add_argument('-r', '--resume', action='store_true',
                          help='resume training from checkpoint or not '
                          + 'Need to set SESSION and EPOCH')
parser_train.add_argument('-te', '--total_epoch', type=int, default=20,
                          help='total number of epochs for training')
parser_train.add_argument('-di', '--display_interval', type=int, default=100,
                          help='number of iterations to display')
parser_train.add_argument('-sd', '--save_dir', default='models',
                          help='directory to save models')
parser_train.add_argument('--vis-off', dest='vis_off', action='store_true',
                          help='turn off visualize training process on plotter')
parser_train.add_argument('-ap', '--add_params', nargs=argparse.REMAINDER,
                          default=[], help='additional parameters')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# create the parser for the test mode
parser_test = subparsers.add_parser('test', formatter_class=formatter,
                                    help='help for TEST mode of network')
add_common_parser_arguments(parser_test)
parser_test.add_argument('-ldd', '--load_dir', default='models',
                         help='directory to load model')
parser_test.add_argument('-ap', '--add_params', nargs=argparse.REMAINDER,
                         default=[], help='additional parameters')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# create the parser for the detect mode
parser_detect = subparsers.add_parser('detect', formatter_class=formatter,
                                      help='help for DETECT mode of network')
add_common_parser_arguments(parser_detect)
parser_detect.add_argument('-ldd', '--load_dir', default='models',
                           help='directory to load model')
parser_detect.add_argument('-imd', '--image_dir', default='images',
                           help='directory to load image files for detection')
parser_detect.add_argument('--vis', action='store_true',
                           help='visualize boxes on image')
parser_detect.add_argument('-ap', '--add_params', nargs=argparse.REMAINDER,
                           default=[], help='additional parameters')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
parser_plot = subparsers.add_parser('plot', formatter_class=formatter,
                                      help='help for PLOT mode of network')
add_common_parser_arguments(parser_plot)
parser_plot.add_argument('-ldd', '--load_dir', default='models',
                         help='directory to load model')
parser_plot.add_argument('-ap', '--add_params', nargs=argparse.REMAINDER,
                           default=[], help='additional parameters')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
parser_cam_train = subparsers.add_parser('cam_train', formatter_class=formatter,
                                      help='help for CAM TRAIN mode of network')
add_common_parser_arguments(parser_cam_train)
parser_cam_train.add_argument('-bs', '--batch_size', type=int, default=None,
                          help='training batch size')
parser_cam_train.add_argument('-lr', '--learning_rate', type=float, default=None,
                          help='training learning rate')
parser_cam_train.add_argument('-r', '--resume', action='store_true',
                          help='resume training from checkpoint or not '
                          + 'Need to set SESSION and EPOCH')
parser_cam_train.add_argument('-te', '--total_epoch', type=int, default=20,
                          help='total number of epochs for training')
parser_cam_train.add_argument('-di', '--display_interval', type=int, default=100,
                          help='number of iterations to display')
parser_cam_train.add_argument('-sd', '--save_dir', default='models',
                          help='directory to save models')
parser_cam_train.add_argument('--vis-off', dest='vis_off', action='store_true',
                          help='turn off visualize training process on plotter')
parser_cam_train.add_argument('-ap', '--add_params', nargs=argparse.REMAINDER,
                           default=[], help='additional parameters')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
parser_cam_test = subparsers.add_parser('cam_test', formatter_class=formatter,
                                    help='help for CAM TEST mode of network')
add_common_parser_arguments(parser_cam_test)
parser_cam_test.add_argument('-ldd', '--load_dir', default='models',
                         help='directory to load model')
parser_cam_test.add_argument('-cam', '--cam_type', default='gradcam',
                         help='which cam to use')
parser_cam_test.add_argument('-ap', '--add_params', nargs=argparse.REMAINDER,
                         default=[], help='additional parameters')

if __name__ == "__main__":
    init(autoreset=True)

    cfg.ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    cfg.DATA_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, 'data'))

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        exit()

    if not args.cuda and torch.cuda.is_available():
        print(Back.YELLOW + Fore.BLACK + 'WARNING! CUDA device is available. '
                                       + 'You can try to run with --cuda flag')
        sys.stdout.write('Continue after 5 seconds... ')
        sys.stdout.flush()
        try:
            for i in range(4, 0, -1):
                time.sleep(1)
                sys.stdout.write('{}... '.format(i))
                sys.stdout.flush()
            print('Run without CUDA.')
        except:
            print('Breaking.')
            exit()
        cfg.CUDA = False
    elif args.cuda and not torch.cuda.is_available():
        print(Back.RED + 'ERROR! CUDA device is unavailable. '
                       + 'You need to run without --cuda flag')
        exit()
    else:
        cfg.CUDA = args.cuda

    print(Back.WHITE + Fore.BLACK + 'Called with args:')
    print(args)

    add_params, err_params = parse_additional_params(args.add_params)
    if len(err_params) > 0:
        print(Back.RED + 'ERROR! Cannot parse next additional parameters:')
        for p in err_params:
            print('\t' + p)
        exit()

    log = logging.getLogger('All_Logs')
    log.setLevel(logging.INFO)
    if args.mode == 'train':
        log_filename = os.path.join(cfg.DATA_DIR, "logs", "wsd_train_sess_{}.log".format(args.session))
        if args.epoch == 1:
            fh = logging.FileHandler(log_filename, mode='w')
        else:
            fh = logging.FileHandler(log_filename, mode='a')
    elif args.mode == 'plot':
        log_filename = os.path.join(cfg.DATA_DIR, "logs", "wsd_stat_sess_{}.log".format(args.session))
        if args.epoch == 1:
            fh = logging.FileHandler(log_filename, mode='w')
        else:
            fh = logging.FileHandler(log_filename, mode='a')
    elif args.mode == 'cam_train':
        log_filename = os.path.join(cfg.DATA_DIR, "logs", "cam_train_sess_{}.log".format(args.session))
        if args.epoch == 1:
            fh = logging.FileHandler(log_filename, mode='w')
        else:
            fh = logging.FileHandler(log_filename, mode='a')
    elif args.mode == 'cam_test':
        log_filename = os.path.join(cfg.DATA_DIR, "logs", "cam_test_sess_{}.log".format(args.session))
        if args.epoch == 1:
            fh = logging.FileHandler(log_filename, mode='w')
        else:
            fh = logging.FileHandler(log_filename, mode='a')
    else:
        log_filename = os.path.join(cfg.DATA_DIR, "logs", "wsd_test_sess_{}.log".format(args.session))
        if args.epoch==1:
            fh = logging.FileHandler(log_filename, mode='w')
        else:
            fh = logging.FileHandler(log_filename, mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    if args.mode == 'train':
        train(dataset_name=args.dataset, net=args.net, batch_size=args.batch_size,
              learning_rate=args.learning_rate, optimizer=args.optimizer,
              lr_decay_step=args.lr_decay_step, lr_decay_gamma=args.lr_decay_gamma,
              pretrain=args.pretrain, resume=args.resume, class_agnostic=args.class_agnostic,
              total_epoch=args.total_epoch, display_interval=args.display_interval,
              session=args.session, epoch=args.epoch, save_dir=args.save_dir,
              vis_off=args.vis_off, visdom_port=args.visdom_port, log=log, mGPU=args.mGPU, add_params=add_params)
    elif args.mode == 'test':
        test(dataset=args.dataset, net=args.net, class_agnostic=args.class_agnostic,
             load_dir=args.load_dir, session=args.session, epoch=args.epoch, log=log,
             add_params=add_params)
    elif args.mode == 'cam_train':
        cam_train(dataset=args.dataset, net=args.net, batch_size=args.batch_size,
              learning_rate=args.learning_rate, resume=args.resume,
              total_epoch=args.total_epoch, display_interval=args.display_interval,
              session=args.session, epoch=args.epoch, save_dir=args.save_dir,
              visdom_port=args.visdom_port, log=log, mGPU=args.mGPU, add_params=add_params)
    elif args.mode == 'cam_test':
        cam_test(dataset=args.dataset, net=args.net, load_dir=args.load_dir, session=args.session,
                 epoch=args.epoch, log=log, cam_type=args.cam_type, add_params=add_params)
    elif args.mode == 'plot':
        hyp_test(dataset=args.dataset, net=args.net, class_agnostic=args.class_agnostic,
             load_dir=args.load_dir, session=args.session, epoch=args.epoch, log=log,
             add_params=add_params)
    else:
        detect(dataset=args.dataset, net=args.net, class_agnostic=args.class_agnostic,
               load_dir=args.load_dir, session=args.session, epoch=args.epoch,
               vis=args.vis, image_dir=args.image_dir, add_params=add_params)

    deinit()
