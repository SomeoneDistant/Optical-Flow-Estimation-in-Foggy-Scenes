import os
import re
import numpy as np
import uuid
from scipy import misc
import numpy as np
from PIL import Image
import sys
import cv2


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:
        endian = '<'
        scale = -scale
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

data, scale = readPFM('/home/yaoyuan/Desktop/viml12/fog_simulation/flyingthings/optical_flow/TRAIN/A/0000/into_future/left/OpticalFlowIntoFuture_0006_L.pfm')
# data = 1 / data
# data = data[:,:,np.newaxis]

# img = cv2.imread('/home/yaoyuan/Desktop/viml12/fog_simulation/flyingthings/frames_cleanpass/TRAIN/A/0000/left/0006.png')
# cv2.imshow('original', img)
# img = np.array(img)

# b = np.random.uniform(1/np.min(data), 3/np.max(data))
# atmosphere = np.exp(-b * data)

# scatter = np.random.uniform(50,150)
# img_fog = img * atmosphere + scatter * (1 - atmosphere)
# cv2.imshow('foggy', img_fog/255)
# cv2.waitKey(delay = 0)
data = data[:,:,0:2]
print(np.max(data[:,:,0]))