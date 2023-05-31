import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchsummary import summary

import numpy as np
import pylab as pl
from tqdm import tqdm
import csv

from datasets import RGZ20k_test

batch_size = 32  # number of samples per mini-batch
imsize = 150  # The image size = 18 x 18 = 324

device = torch.device("cpu")

# -----------------------------------------------------------------------------


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        pl.imshow(npimg, cmap="Greys")
    else:
        pl.imshow(np.transpose(npimg, (1, 2, 0)))

    pl.show()

    return


# -----------------------------------------------------------------------------

transform = transforms.Compose(
    [
        #    transforms.CenterCrop(28),
        #    transforms.RandomRotation(0.,360.),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_data = RGZ20k_test("data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)


# get some random training images
dataiter = iter(train_loader)
images, rgzids, sizes, mbflag = dataiter.next()

# print out RGZ IDs:
print(rgzids)

# print out largest angular sizes (LAS):
print(sizes)

# print out MiraBest flag:
print(mbflag)

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)
