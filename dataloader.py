import torch
import torch.utils.data
import PIL
import os
import numpy as np
from glob import glob

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')

class customData(torch.utils.data.Dataset):
    def __init__(self, datapath, data_transforms=None, loader=My_loader):
        self.files = glob(os.path.join(datapath, 'adult') + '/*.jpg') + glob(os.path.join(datapath, 'kid') + '/*.jpg')
        self.loader = loader
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img = self.loader(self.files[item])
        face = img.crop((0, 0, 224, 224))
        kp_224 = img.crop((224, 0, 448, 224))
        kp_112 = kp_224.resize((112, 112))
        kp_56 = kp_224.resize((56, 56))
        kp_28 = kp_224.resize((28, 28))

        if self.data_transforms is not None:
            face = self.data_transforms(face)
            kp_224 = self.data_transforms(kp_224)
            kp_112 = self.data_transforms(kp_112)
            kp_56 = self.data_transforms(kp_56)
            kp_28 = self.data_transforms(kp_28)
        
        return face, kp_224, kp_112, kp_56, kp_28