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

        if self.data_transforms is not None:
            img = self.data_transforms(img)
        
        return img