import torch
import torch.nn as nn
import numpy as np

def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """ 
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1) 
    image[image == 109] = 0 # erase background (background type 2) 
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1 
    return np.reshape(image.astype(np.float).ravel(), [80,80])
