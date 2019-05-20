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

# H para args
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--gan_type', type=str, default='LogGAN')
parser.add_argument('--gan_w', type=float, default=1)
parser.add_argument('--pix_w', type=float, default=100)
parser.add_argument('--lrG', type=float, default=0.0001)
parser.add_argument('--lrD', type=float, default=0.0001)
parser.add_argument('--lr_decay_step', type=int, default=1)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--random_seed', type=int, default=999)
args = parser.parse_args()

# create file
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

# set model name
model_name = args.dataset + '_' \
         + args.gan_type + '_' \
         + 'gan_w_' + str(args.gan_w) + '_' \
         + 'pix_w_' + str(args.pix_w) + '_' \
         + time.asctime(time.localtime(time.time()))

model = Aging_Model(args, model_name)

model.train()
model.test()

