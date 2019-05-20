import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import os
import numpy
import time
from dataloader import customData
from tensorboardX import SummaryWriter
from utils import *
from networks import *

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

        # Load Discriminator
        print("Loading Discriminator")
        self.D = Discriminator().to(self.device)
        self.D = nn.DataParallel(self.D, list(range(args.ngpu)))
        print_network(self.D)
        
        # Load Generator
        print("Loading Generator")
        self.G = Generator().to(self.device)
        self.G = nn.DataParallel(self.G, list(range(args.ngpu)))
        print_network(self.G)

    def train(self):
        # set random seed
        print(self.model_name)
        setup_seed(self.random_seed)
        print("Set random seed", self.random_seed)

        # prepare data
        datapath = os.path.join(self.dataroot, self.dataset)
        print(datapath)
        data_transforms = transforms.Compose([
            #transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5,.5,.5),std=(.5,.5,.5))
            ])
        trainset = customData(datapath, data_transforms)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        print(len(trainloader))
        log_path = os.path.join(self.log_dir, self.model_name)
        writer = SummaryWriter(log_path)
        for epoch in range(1):
            for iters, data in enumerate(trainloader, 0):
                print(iters)
                face, kp_224, kp_112, kp_56, kp_28 = data
                face = face.to(self.device)
                kp_224 = kp_224.to(self.device)
                kp_112 = kp_112.to(self.device)
                kp_56 = kp_56.to(self.device)
                kp_28 = kp_28.to(self.device)
                self.D(face, kp_224)
                print("Passing D Success")
                self.G(face, kp_224, kp_112, kp_56, kp_28)
                print("Passing G Success")
                if iters % 100 == 0:
                    writer.add_image("face", (face + 1) / 2, iters)
                    writer.add_image("kp_224", (kp_224 + 1) / 2, iters)
                    writer.add_image("kp_112", (kp_112 + 1) / 2, iters)
                    writer.add_image("kp_56", (kp_56 + 1) / 2, iters)
                    writer.add_image("kp_28", (kp_28 + 1) / 2, iters)

        print("Train Success")

    def test(self):
        print("Test Success")
        