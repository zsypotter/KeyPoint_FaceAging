import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def print_network(net):
    num_parameters = 0
    for param in net.parameters():
        num_parameters += param.numel()

    print(net)
    print('Total number of the parameter is: %d' % num_parameters)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            #m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            #m.bias.data.zero_()
        elif isinstance(m, nn.InstanceNorm2d):
            m.weight.data.normal_(1, 0.02)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)