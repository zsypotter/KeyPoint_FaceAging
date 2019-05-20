import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import os
import numpy
import time
from tensorboardX import SummaryWriter
from utils import *

class Aging_Model(object):
    def __init__(self, args, model_name):

        # Path args
        self.dataroot = args.dataroot
        self.dataset = args.dataset
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        # H para args
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.ngpu = args.ngpu
        self.input_size = args.input_size
        self.gan_type = args.gan_type   
        self.gan_w = args.gan_w
        self.pix_w = args.pix_w
        self.lrG = args.lrG
        self.lrD = args.lrD
        self.lr_decay_step = args.lr_decay_step
        self.lr_decay = args.lr_decay
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.random_seed = args.random_seed

        # set model name
        self.model_name = model_name

        # set gpu device
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    def train(self):
        print("Train Success")

    def test(self):
        print("Test Success")
        