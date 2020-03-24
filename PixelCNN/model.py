import os
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, utils

# load MNIST dataset

train = data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                     batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
test = data.DataLoader(datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
                     batch_size=128, shuffle=False, num_workers=1, pin_memory=True)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
        
# define the model
fm = 128
model = nn.Sequential(
    MaskedConv2d('A', 1,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    nn.Conv2d(fm, 256, 1))

model=model.cuda()
optimizer = optim.Adam(model.parameters())

# Train loop

for epoch in range(30):
    # train
    err_train = []
    time_tr = time.time()
    model.train(True)
    for input, _ in tr:
        input = input.cuda()
        target = (input.data[:,0] * 255).long()
        loss = F.cross_entropy(model(input), target)
        err_train.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    trainTime=time.time() - time_tr

    # compute error on test set
    err_test = []
    time_te = time.time()
    model.train(False)
    for input, _ in te:
        input = input.cuda()
        target = (input.data[:,0] * 255).long()
        loss = F.cross_entropy(model(input), target)
        err_test.append(loss.item())

    print("Epoch: {} , Train error = {} , Test error: {} ,time Elapsed = {}".format( epoch+1,np.mean(err_train),np.mean(err_test),trainTime)   )

# save the model
# torch.save(model.state_dict(), "last.pt")
