import os
import sys

import torch
import numpy as np
from utils import make_multipleof

# Let's define MRIDataset class
class MRIDataset(torch.utils.data.Dataset):
  def __init__(self, rootDir, length):
    self.rootDir = rootDir
    self.length = length
  
  def __len__(self):
    return self.length

  def __getitem__(self, index):
    imgName = "{:>03d}_imgs.npy".format(index+1)
    labelName = "{:>03d}_seg.npy".format(index+1)
    img = np.load(os.path.join(self.rootDir, imgName))
    orig_dims = img.shape[1:]
    label = np.load(os.path.join(self.rootDir, labelName))
    label = label.astype(int)
    img = make_multipleof(img, 16)
    img, label = torch.from_numpy(img), torch.from_numpy(label)
    return img, label, orig_dims

    