import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
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

        # set criterion
        self.mse_criterion = nn.MSELoss().to(self.device)
        self.bce_criterion = nn.BCELoss().to(self.device)

        # set optimizer
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))
        self.G_scheduler = torch.optim.lr_scheduler.StepLR(self.G_optimizer, self.lr_decay_step, gamma=self.lr_decay)
        self.D_scheduler = torch.optim.lr_scheduler.StepLR(self.D_optimizer, self.lr_decay_step, gamma=self.lr_decay)

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
        
        # train loop

        print('Start Training Loop...')
        iters_num = len(trainloader)
        log_path = os.path.join(self.log_dir, self.model_name)
        writer = SummaryWriter(log_path)
        self.G.train()
        self.D.train()
        for epoch in range(self.epoch):

            self.G_scheduler.step()
            self.D_scheduler.step()
            for iters, data in enumerate(trainloader, 0):

                start_time = time.clock()
                # load data
                face, kp_224, kp_112, kp_56, kp_28 = data
                face = face.to(self.device)
                kp_224 = kp_224.to(self.device)
                kp_112 = kp_112.to(self.device)
                kp_56 = kp_56.to(self.device)
                kp_28 = kp_28.to(self.device)
                b_size = face.size(0)
                real_label = torch.ones((b_size, 1, 28, 28)).to(self.device)
                fake_label = torch.zeros((b_size, 1, 28, 28)).to(self.device)

                # update G
                self.G.zero_grad()
                fake_face = self.G(face, kp_224, kp_112, kp_56, kp_28)
                fake_face_D = self.D(fake_face, kp_224)
                if self.gan_type == 'LogGAN':
                    fake_face_D = F.sigmoid(fake_face_D)
                    errG_fake = self.bce_criterion(fake_face_D, real_label)
                else:
                    errG_fake = self.mse_criterion(fake_face_D, real_label)
                gan_loss = errG_fake
                pix_loss = self.mse_criterion(face, fake_face) / (self.input_size * self.input_size * 3)
                G_loss = self.gan_w * gan_loss + self.pix_w * pix_loss
                G_loss.backward()
                self.G_optimizer.step()

                # update D
                self.D.zero_grad()
                face_D = self.D(face, kp_224)
                if self.gan_type == 'LogGAN':
                    face_D = F.sigmoid(face_D)
                    errD_real = self.bce_criterion(face_D, real_label)
                else:
                    errD_real = self.mse_criterion(face_D, real_label)
                fake_face_D = self.D(fake_face.detach(), kp_224)
                if self.gan_type == 'LogGAN':
                    fake_face_D = F.sigmoid(fake_face_D)
                    errD_fake = self.bce_criterion(fake_face_D, fake_label)
                else:
                    errD_fake = self.mse_criterion(fake_face_D, fake_label)       
                D_loss = (errD_real + errD_fake) / 2
                D_loss.backward()
                self.D_optimizer.step()

                end_time = time.clock()
                print('epochs: [{}/{}], iters: [{}/{}], per_iter {:.4f}, G_loss: {:.4f}, D_loss: {:.4f}, lrG: {:.8f}, lrD: {:.8f}'.format(epoch, self.epoch, iters, iters_num, end_time - start_time, errG_fake.item(), D_loss.item(), self.G_optimizer.param_groups[0]['lr'], self.D_optimizer.param_groups[0]['lr']))

                if iters % 100 == 0:
                    writer.add_image("face", (face + 1) / 2, iters + epoch * iters_num)
                    writer.add_image("fake_face", (fake_face + 1) / 2, iters + epoch * iters_num)
                    writer.add_scalar('errD_real', errD_real, iters + epoch * iters_num)
                    writer.add_scalar('errD_fake', errD_fake, iters + epoch * iters_num)
                    writer.add_scalar('errG_fake', errG_fake, iters + epoch * iters_num)

        print("Train Success")

    def test(self):
        print("Test Success")
        