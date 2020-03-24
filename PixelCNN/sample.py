import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, utils


model=model.load_state_dict(torch.load("last.pt")).cuda()

num_samples=128
sample = torch.Tensor(num_samples, 1, 28, 28).cuda()
sample.fill_(0)
model.train(False)
for i in range(28):
    for j in range(28):
        output = model(sample)
        probs = F.softmax(output[:, :, i, j]).data
        sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.


# to save the samples on disk
#utils.save_image(sample, 'samples.png', nrow=12, padding=0)
