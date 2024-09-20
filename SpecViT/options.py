import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # base options
        self.initialized = True
        self.parser.add_argument('--dataroot', type=str, default="/data/shah/projects/spms_track/experiments/beads/488beads/", help='path to images')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--model', type=str, default='MDD_ViT',  help='# cnn, ViT, ViT_skip')
        self.parser.add_argument('--pos_manner', type=str, default='learned' )

        self.parser.add_argument('--pretrained_filename', type=str, default='MDD_ViT-learned-latest.ckpt')
        self.parser.add_argument('--logger', type=str, default='True')

        # dataset options
        self.parser.add_argument('--N_per_C', type=int, default=720, help='N per centroid')
        
        # ViT parameters

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            self.opt.gpu_ids.append(int(str_id))
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # # save to the disk
        # expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.model)
        # if not os.path.exists(expr_dir):
        #     os.makedirs(expr_dir)
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write('------------ Options -------------\n')
        #     for k, v in sorted(args.items()):
        #         opt_file.write('%s: %s\n' % (str(k), str(v)))
        #     opt_file.write('-------------- End ----------------\n')
        return self.opt


