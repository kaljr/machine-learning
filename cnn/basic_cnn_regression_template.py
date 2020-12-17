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
import time
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Pool, Process, set_start_method
from torchsummary import summary

# For cuda/multiprocessing compatability
try:
  set_start_method('spawn', force=True)
except RuntimeError:
  pass
  

# Set the device that will be used to train the network
cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")

if cuda_available:
  print('using gpu:', torch.cuda.get_device_name(0))
else:
  print('using', device)

# Define file constants (change to suit project)
ROOT_DATA_PATH = ("D:/some-dir")

# These expect a directory of json files that are named the same as the associated training image file
# The json data will be read and the associated image file will then be read and associate with the data
# In this case the json data is expected to be a list of landmark points, [{'x': 0, 'y': 0}, {'x': 0, 'y': 0}]...
TRAINING_DATA_DIR = ("train/")
VALIDATION_DATA_DIR = ("validate/") # validation data will not be seen by the network while training
SAVE_MODEL = True
SAVE_MODEL_PATH = './'

# Training variables
SOURCE_IMG_H = 1080 # native resolution of image files
SOURCE_IMG_W = 1920
INPUT_IMG_H = 432 # desired resolution of images to be put into the CNN
INPUT_IMG_W = 768
NUM_TRAINING_EPOCHS = 1
BATCH_SIZE = 4

# General Functions
def json_read(json_path):
  with open(json_path, "r") as read_file:
    data = json.load(read_file)

  return data

def data_to_tensor(data):
  """
  turn pts data from [{'x': 0, 'y': 0} ...] into one-dimensional tensor
  containing all points [pt1x, pt1y, pt2x, pt2y, ...]
  """
  pts_vector = []

  for pt in data:
    pts_vector.append(pt['x'])
    pts_vector.append(pt['y'])
  
  return torch.tensor(pts_vector)

def build_file_info(data_file_names):
  """ create dictionary of folder names and file (image/data) numbers """
  dirs = {}

  for file_name in data_file_names:
    dir = file_name[:-10] # assumes the file name is like 'somefile_0002.json'
    fileNum = file_name[-9:-5]

    if (dir == '.ipynb_c'): #skip a hidden folder created by pytorch
      continue

    if dir in dirs:
      dirs[dir].append(fileNum)
    else:
      dirs[dir] = [fileNum]

  return dirs

def resizeImageTensor(image_tensor, h, w):
  # interpolate only deals with batches, so must use unsqueeze and 0th index to appease it
  return torch.nn.functional.interpolate(torch.unsqueeze(image_tensor, 0), size=(h, w))[0]

def scalePts(pts, scale):
  # Expects the pts to be like: [{'x': 0, 'y': 0} ...]
  for i,pt in enumerate(pts):
    pts[i]['x'] = pt['x'] * scale
    pts[i]['y'] = pt['y'] * scale

  return pts

def build_image_info(file_info, root_dir, data_dir):
  """ Create the associations between json data and images """
  img_info = [];

  for name in file_info:
    for img_num in file_info[name]:
      pts = json_read(root_dir + data_dir + name + '_' + img_num + '.json')
      pts = scalePts(pts, (INPUT_IMG_H/SOURCE_IMG_H))
  
      img_info.append({
          'name': name,
          'num': img_num,
          'img_path': root_dir + name + '/' + name + '_' + img_num + '.jpg',
          'pt_values': data_to_tensor(pts)
      })

  return img_info

class ImageLandmarksDataset(Dataset):
  """ Create the dataset that will be used by the DataLoader to access the images and associated training data """

  def __init__(self, file_info, transform=None):
    """
      Args:
        file_info (dict): image path and point data {img_path: <string>, pt_values: <tensor>}
    """
    self.transform = transform
    self.file_info = file_info

  def __len__(self):
    """ must return the size of the dataset """
    return len(self.file_info)

  def __getitem__(self, i):
    """ must ensure that dataset[i] can be used to get the ith sample"""
    file_info = self.file_info[i]
    tensor_type = torch.FloatTensor if device == 'cpu' else torch.cuda.FloatTensor

    # must be cast to float to perform training and loss calculations
    img = resizeImageTensor( img_to_tensor(file_info['img_path']), INPUT_IMG_H, INPUT_IMG_W ).type(tensor_type).to(device)
    pt_values = file_info['pt_values'].type(tensor_type).to(device)

    sample = {'image': img, 'pt_values': pt_values}

    if self.transform:
      sample = self.transform(sample)

    return sample
  
  
# Set up data for training/validation
training_data_file_names = listdir(ROOT_DATA_PATH + TRAINING_DATA_DIR)
validation_data_file_names = listdir(ROOT_DATA_PATH + VALIDATION_DATA_DIR)

training_file_info = build_file_info(training_data_file_names)
training_img_info = build_image_info(training_file_info, ROOT_DATA_PATH, TRAINING_DATA_DIR)

training_dataset = ImageCornersDataset(training_img_info)

# sometimes data can take a while to load, let user know this completed
print('\nSuccessfully built training info. Number of traning samples:' + len(training_dataset) + '\n\n')

train_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# Create CNN
net = nn.Sequential(
    nn.Conv2d(3, 12, kernel_size=50, stride=4), nn.BatchNorm2d(12), nn.ReLU(),
    nn.MaxPool2d(kernel_size=10, stride=2),
    nn.Conv2d(12, 24, kernel_size=20), nn.BatchNorm2d(24), nn.ReLU(),
    nn.MaxPool2d(kernel_size=4, stride=2),
    nn.Conv2d(24, 48, kernel_size=4), nn.BatchNorm2d(48), nn.ReLU(),
    nn.MaxPool2d(kernel_size=4, stride=2),
    nn.Conv2d(48, 96, kernel_size=4), nn.BatchNorm2d(96), nn.ReLU(),
    nn.MaxPool2d(kernel_size=4, stride=2),
    nn.Flatten(),
    nn.Linear(19008, 120), nn.BatchNorm1d(120), nn.ReLU(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.ReLU(),
    nn.Linear(84, 8)
)
net.to(device) # must move network to device being used

# This will print a summary of the CNN that includes number of network parameters, output dimensions, etc
summary(model,(3, INPUT_IMG_H, INPUT_IMG_W)) # second argument is input dimensions ie: (3,512,1024)

# Set up loss function and optimization algorithm
loss_fn = nn.MSELoss() # MSE is good for regression (not classification)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-08)


# Train the network
for epoch in range(NUM_TRAINING_EPOCHS):  # loop over the dataset multiple times
    print('Training epoch', epoch + 1, ':\n\n')

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels/values]
        image, pt_values = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(data['image'])
        loss = loss_fn(outputs, data['pt_values'])
        loss.backward()
        optimizer.step()

        # print statistics
        batch_loss = loss.item()
        running_loss += batch_loss
        
        if i % 5 == 4:    # print every 5 mini-batches
            print('[%d, %5d] batch_loss: %.3f loss: %.3f' %
                  (epoch + 1, i + 1, batch_loss, running_loss / 5))
            running_loss = 0.0


# Save the trained model data to file for later use/reference
if SAVE_MODEL:
  model_state = net.state_dict()
  model_file_name = 'cnn-landmark_' + int(round(time.time())) + '.pth'
  torch.save(model_state, SAVE_MODEL_PATH + model_file_name) # save model coefficients to file
  print('/nModel saved/n')

print('Finished Training')

##### Validation Section ######
# This loads the network stucture, and then the saved coefficients
# net = Net() #
# net.load_state_dict(torch.load(SAVE_MODEL_PATH + model_file_name))


# Set up data for validation
# validation_file_info = build_file_info(validation_data_file_names)
# validation_img_info = build_image_info(validation_file_info, ROOT_DATA_PATH, VALIDATION_DATA_DIR)
# validation_dataset = ImageLandmarksDataset(validation_img_info)
# corners_validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=0)

# print('\nSuccessfully built validation info. Number of traning samples:' + len(validation_dataset) + '\n\n')


# Perform inference for validation
# running_loss = 0.0
# for i, data in enumerate(validation_loader, 0):
#     # get the inputs; data is a list of [inputs, labels]
#     image, pt_values = data

#     # forward
#     outputs = net(data['image'])
#     loss = loss_fn(outputs, data['pt_values'])

#     # print statistics
#     batch_loss = loss.item()
#     running_loss += batch_loss

#     if i % 5 == 4:    # print every 5 mini-batches
#       print('[%d, %5d] batch_loss: %.3f loss: %.3f' %
#             (0, i + 1, batch_loss, running_loss / BATCH_SIZE))
#       running_loss = 0.0

# print('Finished Validation')
