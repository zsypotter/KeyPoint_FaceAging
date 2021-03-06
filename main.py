import argparse
import os
import torch
import time
from models import Aging_Model

parser = argparse.ArgumentParser()
# Path args
parser.add_argument('--dataroot', type=str, default='/data2/zhousiyu/dataset/')
parser.add_argument('--dataset', type=str, default='K2A')
parser.add_argument('--save_dir', type=str, default='dict/')
parser.add_argument('--result_dir', type=str, default='results/')
parser.add_argument('--log_dir', type=str, default='runs/')
parser.add_argument('--eval_model', type=str, default='/data2/zhousiyu/workspace/FaceAging/KeyPoint_FaceAging/dict/rLinear_32_30_1.pth')
parser.add_argument('--eval', type=int, default=0)

# Model discription
parser.add_argument('--discription', type=str, default='')

# H para args
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--gan_type', type=str, default='MseGAN')
parser.add_argument('--gan_w', type=float, default=1)
parser.add_argument('--pix_w', type=float, default=10)
parser.add_argument('--use_perceptual', type=int, default=1)
parser.add_argument('--feat_w', type=float, default=1)
parser.add_argument('--style_w', type=float, default=5)
parser.add_argument('--perceptual_w', type=float, default=0.001)
parser.add_argument('--lrG', type=float, default=0.0001)
parser.add_argument('--lrD', type=float, default=0.0001)
parser.add_argument('--lr_decay_step', type=int, default=1)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--random_seed', type=int, default=999)
args = parser.parse_args()

# set model name
if args.use_perceptual == 1:
    model_name = args.dataset + '_' \
            + args.gan_type + '_' \
            + 'gan_w_' + str(args.gan_w) + '_' \
            + 'pix_w_' + str(args.pix_w) + '_' \
            + 'feat_w_' + str(args.feat_w) + '_' \
            + 'style_w_' + str(args.style_w) + '_' \
            + 'perceptual_w' + str(args.perceptual_w) + '_' \
            + time.asctime(time.localtime(time.time())) + '_' \
            + args.discription
else:
    model_name = args.dataset + '_' \
            + args.gan_type + '_' \
            + 'gan_w_' + str(args.gan_w) + '_' \
            + 'pix_w_' + str(args.pix_w) + '_' \
            + time.asctime(time.localtime(time.time())) + '_' \
            + args.discription

# create file
if not os.path.exists(os.path.join(args.save_dir, model_name)):
    os.makedirs(os.path.join(args.save_dir, model_name))

if not os.path.exists(os.path.join(args.result_dir, args.eval_model.split('/')[-1].split('.')[0])):
    os.makedirs(os.path.join(args.result_dir, args.eval_model.split('/')[-1].split('.')[0]))



model = Aging_Model(args, model_name)

if args.eval == 0:
    model.train()
else:
    model.test()

