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
        self.use_perceptual = args.use_perceptual
        self.feat_w = args.feat_w
        self.style_w = args.style_w
        self.perceptual_w = args.perceptual_w
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

        # Load VGG19
        if self.use_perceptual == 1:
            print("Loading VGG19")
            self.VGG19 = VGG19().to(self.device)
            self.VGG19 = nn.DataParallel(self.VGG19, list(range(args.ngpu)))
            print_network(self.VGG19)

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

        # define mean and std for VGG
        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
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

                # gan loss
                fake_face_D = self.D(fake_face, kp_224)
                if self.gan_type == 'LogGAN':
                    fake_face_D = F.sigmoid(fake_face_D)
                    errG_fake = self.bce_criterion(fake_face_D, real_label)
                else:
                    errG_fake = self.mse_criterion(fake_face_D, real_label)
                gan_loss = errG_fake

                # pix loss
                pix_loss = self.mse_criterion(face, fake_face) / (self.input_size * self.input_size * 3)

                # perceptual loss
                if self.use_perceptual == 1:
                    fake_face_vgg = (fake_face + 1) / 2
                    face_vgg = (face + 1) / 2
                    fake_face_vgg = (fake_face_vgg - vgg_mean) / vgg_std
                    face_vgg = (face_vgg - vgg_mean) / vgg_std
                    f_relu1_2, f_relu2_2, f_relu3_3, f_relu4_3 = self.VGG19(fake_face_vgg)
                    relu1_2, relu2_2, relu3_3, relu4_3 = self.VGG19(face_vgg)
                    f_g1 = gram_matrix(f_relu1_2)
                    f_g2 = gram_matrix(f_relu2_2)
                    f_g3 = gram_matrix(f_relu3_3)
                    f_g4 = gram_matrix(f_relu4_3)
                    g1 = gram_matrix(relu1_2)
                    g2 = gram_matrix(relu2_2)
                    g3 = gram_matrix(relu3_3)
                    g4 = gram_matrix(relu4_3)

                    feat_loss = self.mse_criterion(f_g3, g3)
                    style_loss = self.mse_criterion(f_g1, g1) + self.mse_criterion(f_g2, g2) + self.mse_criterion(f_g3, g3) + self.mse_criterion(f_g4, g4)
                    perceptual_loss = self.feat_w * feat_loss + self.style_w * style_loss

                if self.use_perceptual == 1:
                    G_loss = self.gan_w * gan_loss + self.pix_w * pix_loss + self.perceptual_w * perceptual_loss
                else:
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
                if self.use_perceptual == 1:
                    print('epochs: [{}/{}], iters: [{}/{}], per_iter {:.4f}, G_loss: {:.4f}, D_loss: {:.4f}, pix_loss: {:.4f}, feat_loss: {:.4f}, style_loss: {:.4f}'.format(epoch, self.epoch, iters, iters_num, end_time - start_time, errG_fake.item(), D_loss.item(), pix_loss.item(), feat_loss.item(), style_loss.item()))
                else:
                    print('epochs: [{}/{}], iters: [{}/{}], per_iter {:.4f}, G_loss: {:.4f}, D_loss: {:.4f}, pix_loss: {:.4f}'.format(epoch, self.epoch, iters, iters_num, end_time - start_time, errG_fake.item(), D_loss.item(), pix_loss.item()))
                if iters % 100 == 0:
                    writer.add_image("face", (face + 1) / 2, iters + epoch * iters_num)
                    writer.add_image("fake_face", (fake_face + 1) / 2, iters + epoch * iters_num)
                    writer.add_scalar('errD_real', errD_real, iters + epoch * iters_num)
                    writer.add_scalar('errD_fake', errD_fake, iters + epoch * iters_num)
                    writer.add_scalar('errG_fake', errG_fake, iters + epoch * iters_num)
                    writer.add_scalar('pix_loss', pix_loss, iters + epoch * iters_num)
                    if self.use_perceptual == 1:
                        writer.add_scalar('feat_loss', feat_loss, iters + epoch * iters_num)
                        writer.add_scalar('style_loss', feat_loss, iters + epoch * iters_num)

        print("Train Success")

    def test(self):
        print("Test Success")
        