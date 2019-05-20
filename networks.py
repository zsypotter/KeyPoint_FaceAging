import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 224
            nn.Conv2d(6, 128, 3, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),
            # 112
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),
            # 56
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2),
            # 28
            nn.Conv2d(512, 1, 3, 1, 1),
            # 28
        )
    
    def forward(self, face, kp_224):
        x = torch.cat((face, kp_224), 1)
        x = self.main(x)

        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 224
        self.e_conv1 = nn.Conv2d(3, 64, 9, 2, 4)
        self.e_in1 = nn.InstanceNorm2d(64, affine=True)
        self.e_r1 = nn.ReLU(True)
        # 112
        self.e_conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.e_in2 = nn.InstanceNorm2d(128, affine=True)
        self.e_r2 = nn.ReLU(True)
        # 56
        self.e_conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.e_in3 = nn.InstanceNorm2d(256, affine=True)
        self.e_r3 = nn.ReLU(True)
        # 28
        self.l_l1 = nn.Linear(28 * 28 * 256, 50)
        self.l_t1 = nn.Tanh() 
        self.l_l2 = nn.Linear(50, 28 * 28 * 128)
        self.l_r2 = nn.ReLU(True)

        # 28
        self.r1_conv1 = nn.Conv2d(131, 128, 3, 1, 1)
        self.r1_in1 = nn.InstanceNorm2d(128, affine=True)
        self.r1_r1 = nn.ReLU(True)
        self.r1_conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.r1_in2 = nn.InstanceNorm2d(128, affine=True)
        self.r1_r2 = nn.ReLU(True)
        # 28
        self.r2_conv1 = nn.Conv2d(131, 128, 3, 1, 1)
        self.r2_in1 = nn.InstanceNorm2d(128, affine=True)
        self.r2_r1 = nn.ReLU(True)
        self.r2_conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.r2_in2 = nn.InstanceNorm2d(128, affine=True)
        self.r2_r2 = nn.ReLU(True)
        # 28
        self.r3_conv1 = nn.Conv2d(131, 128, 3, 1, 1)
        self.r3_in1 = nn.InstanceNorm2d(128, affine=True)
        self.r3_r1 = nn.ReLU(True)
        self.r3_conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.r3_in2 = nn.InstanceNorm2d(128, affine=True)
        self.r3_r2 = nn.ReLU(True)
        # 28
        self.r4_conv1 = nn.Conv2d(131, 128, 3, 1, 1)
        self.r4_in1 = nn.InstanceNorm2d(128, affine=True)
        self.r4_r1 = nn.ReLU(True)
        self.r4_conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.r4_in2 = nn.InstanceNorm2d(128, affine=True)
        self.r4_r2 = nn.ReLU(True)
        # 28

        # 28
        self.d_conv1 = nn.Conv2d(131, 512, 3, 1, 1)
        self.d_ps1 = nn.PixelShuffle(2)
        self.d_in1 = nn.InstanceNorm2d(128)
        self.d_r1 = nn.ReLU(True)
        # 56
        self.d_conv2 = nn.Conv2d(131, 512, 3, 1, 1)
        self.d_ps2 = nn.PixelShuffle(2)
        self.d_in2 = nn.InstanceNorm2d(128)
        self.d_r2 = nn.ReLU(True)
        # 112
        self.d_conv3 = nn.Conv2d(131, 512, 3, 1, 1)
        self.d_ps3 = nn.PixelShuffle(2)
        self.d_in3 = nn.InstanceNorm2d(128)
        self.d_r3 = nn.ReLU(True)
        # 224
        self.d_conv4 = nn.Conv2d(131, 3, 9, 1, 4)
        self.d_t4 = nn.Tanh()
        # 224
    
    def forward(self, face, kp_224, kp_112, kp_56, kp_28):

        # 224
        x = self.e_conv1(face)
        x = self.e_in1(x)
        x = self.e_r1(x)
        # 112
        x = self.e_conv2(x)
        x = self.e_in2(x)
        x = self.e_r2(x)
        # 56
        x = self.e_conv3(x)
        x = self.e_in3(x)
        x = self.e_r3(x)
        # 28

        x = x.view(-1, 28 * 28 * 256)
        x = self.l_l1(x)
        x = self.l_t1(x)
        x = self.l_l2(x)
        x = self.l_r2(x)

        # 28
        x = x.view(-1, 128, 28, 28)
        x_r = x
        x = torch.cat((x, kp_28), 1)
        x = self.r1_conv1(x)
        x = self.r1_in1(x)
        x = self.r1_r1(x)
        x = self.r1_conv2(x)
        x = self.r1_in2(x)
        x = x_r + x
        x = self.r1_r2(x)

        x_r = x
        x = torch.cat((x, kp_28), 1)
        x = self.r2_conv1(x)
        x = self.r2_in1(x)
        x = self.r2_r1(x)
        x = self.r2_conv2(x)
        x = self.r2_in2(x)
        x = x_r + x
        x = self.r2_r2(x)

        x_r = x
        x = torch.cat((x, kp_28), 1)
        x = self.r3_conv1(x)
        x = self.r3_in1(x)
        x = self.r3_r1(x)
        x = self.r3_conv2(x)
        x = self.r3_in2(x)
        x = x_r + x
        x = self.r3_r2(x)

        x_r = x
        x = torch.cat((x, kp_28), 1)
        x = self.r4_conv1(x)
        x = self.r4_in1(x)
        x = self.r4_r1(x)
        x = self.r4_conv2(x)
        x = self.r4_in2(x)
        x = x_r + x
        x = self.r4_r2(x)
        # 28

        # 28
        x = torch.cat((x, kp_28), 1)
        x = self.d_conv1(x)
        x = self.d_ps1(x)
        x = self.d_in1(x)
        x = self.d_r1(x)
        # 56
        x = torch.cat((x, kp_56), 1)
        x = self.d_conv2(x)
        x = self.d_ps2(x)
        x = self.d_in2(x)
        x = self.d_r2(x)
        # 112
        x = torch.cat((x, kp_112), 1)
        x = self.d_conv3(x)
        x = self.d_ps3(x)
        x = self.d_in3(x)
        x = self.d_r3(x)
        # 224
        x = torch.cat((x, kp_224), 1)
        x = self.d_conv4(x)
        out = self.d_t4(x)
        
        return out
