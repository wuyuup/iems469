import numpy as np
import torch
import os

def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """ 
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1) 
    image[image == 109] = 0 # erase background (background type 2) 
    image[image != 0] = 1 # everything else just set to 1
    return np.reshape(image.astype(np.float32).ravel(), [80,80])

    
mspacman_color = 210 + 164 + 74
def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127 
    return img.reshape(88, 80, 1)


def preprocess_state(state, env):
    if env.unwrapped.spec.id == 'Breakout-v0':
        return torch.from_numpy(preprocess(state)).unsqueeze(0).unsqueeze(0)
    elif env.unwrapped.spec.id == 'MsPacman-v0':
        state = preprocess_observation(state)
        state = state.reshape([88, 80]) # sb
        return torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
