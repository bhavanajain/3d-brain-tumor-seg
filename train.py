import os
import sys

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from dataset import MRIDataset
from unet import Modified3DUNet
from utils import dice_loss, dice_score

import subprocess
from datetime import date

# Fix seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MODEL_DIR = 'models/'
EXPERIMENT_NAME = '3DUNet'
EXPERIMENT_ID = None
TOTAL_EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 1
LR = 1e-3
LAST_SAVED_MODEL = None
DEBUG = False

if EXPERIMENT_ID is None:
  LAST_COMMIT = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')[:7]
  DATE = date.today().strftime("%b_%d")
  EXPERIMENT_ID = "{}_{}_{}".format(EXPERIMENT_NAME, LAST_COMMIT, DATE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on {}".format(device))

if DEBUG:
  trainDir = '/home/bmjain2/brain-tumor-seg/tinydata/train'
  trainLen = 5
  valDir = '/home/bmjain2/brain-tumor-seg/tinydata/validation'
  valLen = 2
else:
  trainDir = "/home/bmjain2/brain-tumor-seg/data_pub/train"
  trainLen = 204
  valDir = "/home/bmjain2/brain-tumor-seg/data_pub/validation"
  valLen = 68

# Datasets and Dataloaders
trainset = MRIDataset(rootDir=trainDir, length=trainLen)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

valset = MRIDataset(rootDir=valDir, length=valLen)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

net = Modified3DUNet(in_channels=4, n_classes=4, base_n_filter = 8).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=LR)

if not os.path.exists(os.path.join(MODEL_DIR, EXPERIMENT_ID)):
  os.makedirs(os.path.join(MODEL_DIR, EXPERIMENT_ID))
  print("Saving models from this run to {}".format(os.path.join(MODEL_DIR, EXPERIMENT_ID)))

# Load last checkpoint if available
start_epoch = 0
if LAST_SAVED_MODEL is not None:
  print('Loading last saved model ...')
  checkpoint = torch.load(os.path.join(MODEL_DIR, EXPERIMENT_ID, LAST_SAVED_MODEL))
  net.load_state_dict(checkpoint['net_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  prev_epochs, prev_loss, prev_score = checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_score']
  print('prev epochs = {}, train loss = {:.3f}, val score = {:.3f}'.format(prev_epochs, prev_loss, prev_score))
  start_epoch = checkpoint['epoch'] + 1

num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Number of trainable parameters = ", num_params)

loss_history = []
# best score on validation data
best_score, best_epoch = None, None

for epoch in range(start_epoch, TOTAL_EPOCHS):
    # Train the network
    net = net.train()
    train_loss = 0.0
    for data in tqdm(trainloader, position=0, leave=True):
      # get the inputs; mri_scans [B, 4, H, W, D] and target [B, H, W, D]
      # each target voxel has a value between 0 and 3
      mri_scan, target, orig_dims = data[0].to(device), data[1].to(device), data[2]

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      # logits [B, 4, H, W, D]
      logits = net(mri_scan)
      output = F.softmax(logits, dim=1)

      # unpad output
      h, w, d = orig_dims
      output = output[:, :, :h, :w, :d]

      loss = dice_loss(output, target)
      train_loss += loss.item()

      loss.backward()
      optimizer.step()

      torch.cuda.empty_cache()

    train_loss = train_loss / len(trainloader)
    loss_history.append(train_loss)

    # Evaluate the network
    net = net.eval()
    with torch.no_grad():
      val_score = 0.0
      for data in tqdm(valloader, position=0, leave=True):
        # get the inputs; mri_scans [B, 4, H, W, D] and target [B, H, W, D]
        # each target voxel has a value between 0 and 3
        mri_scan, target, orig_dims = data[0].to(device), data[1].to(device), data[2]

        # forward + backward + optimize
        # logits [B, 4, H, W, D]
        logits = net(mri_scan)
        output = F.softmax(logits, dim=1)

        h, w, d = orig_dims
        output = output[:, :, :h, :w, :d]

        score = dice_score(output, target)
        val_score += score.item()

        torch.cuda.empty_cache()

      val_score = val_score / len(valloader)

    MODEL_NAME = "epoch_{}_train_loss_{:.3f}_val_score_{:.3f}".format(epoch, train_loss, val_score)

    torch.save(
        {
          'epoch': epoch,
          'net_state_dict': net.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'train_loss': train_loss,
          'val_score': val_score,
        }, os.path.join(MODEL_DIR, EXPERIMENT_ID, MODEL_NAME))

    print("[Epoch %d] train loss: %.3f, val score: %.3f" % (epoch, train_loss, val_score))

    # Replace best model if validation score improves
    if best_score is None or best_score < val_score:
      best_score, best_epoch = val_score, epoch
      model_path = os.path.join(MODEL_DIR, EXPERIMENT_ID, 'best_tumor_segmentor.pth')
      torch.save(net.state_dict(), model_path)
    
    if epoch - best_epoch > PATIENCE:
      # stop training if score hasn't improved in the last PATIENCE number of epochs
      print("Stopping early at epoch {}, score hasn't improved since {}".format(epoch, best_epoch))
      break