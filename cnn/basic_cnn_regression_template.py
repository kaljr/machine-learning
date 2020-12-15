# This is a basic convolutional neural network written in PyTorch
# It seeks to lay out the general steps to creating, training, loading, and saving a CNN
# It also seeks to serve as a template and reference for some of the basics/minutiae around getting it all done
# This includes things like running the network on a GPU, creating Datasets and using DataLoaders, etc
# It therefore does not use pre-trained networks, or ready-made datasets

# This particular network is a regression CNN;
# i.e. one that seeks to achieve explicit numerical values that do not represent classes
# This network will try to locate some numerical (x,y) landmarks in an image

from matplotlib import pyplot as plt

import numpy as np
import torch as torch
import torch.nn as nn

import torch.optim as optim
import json
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image as img_to_tensor
from os import listdir
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Pool, Process, set_start_method

# for cuda/multiprocessing compatability
try:
  set_start_method('spawn', force=True)
except RuntimeError:
  pass
  

# set the device that will be used to train the network
cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")

if cuda_available:
  print('using gpu:', torch.cuda.get_device_name(0))
else:
  print('using', device)
