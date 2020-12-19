import math
import numpy as np
import torch

def make_multipleof(img, multipleof=16):
    hi, wi, di = img.shape[1:]
    
    hi_pad, wi_pad, di_pad = 0, 0, 0
    if hi % multipleof != 0:
        hi_pad = math.ceil(hi/multipleof) * multipleof - hi
    if wi % multipleof != 0:
        wi_pad = math.ceil(wi/multipleof) * multipleof - wi
    if di % multipleof != 0:
        di_pad = math.ceil(di/multipleof) * multipleof - di
        
    img = np.pad(img, pad_width=((0, 0), (0, hi_pad), (0, wi_pad), (0, di_pad)), mode='constant', constant_values=0)
    return img

def dice_loss(output, target):
    # output [B, 4, H, W (, D)] and target [B, H, W (, D)]
    smooth = 1.0
    num_classes = output.shape[1]
    for i in range(num_classes):
        output_i = output[:, i, ...]
        target_i = (target == i).type(torch.uint8)
        intersection = (output_i * target_i).sum()
        dice_score = (2 * intersection + smooth) / (output_i.sum() + target_i.sum() + smooth)
        dice_loss = 1 - dice_score
        if i == 0:
            total_loss = dice_loss
        else:
            total_loss += dice_loss
    total_loss = total_loss / num_classes
    return total_loss

def dice_score(output, target):
    # output [B, 4, H, W (, D)] and target [B, H, W (, D)]
    num_classes = output.shape[1]
    output = output.argmax(dim=1)
    smooth = 1.0
    for i in range(num_classes):
        output_i = (output == i).type(torch.uint8)
        target_i = (target == i).type(torch.uint8)
        intersection = (output_i * target_i).sum()
        dice_score = (2 * intersection + smooth) / (output_i.sum() + target_i.sum() + smooth)

        if i == 0:
            total_score = dice_score
        else:
            total_score += dice_score
    total_score = total_score / num_classes
    return total_score