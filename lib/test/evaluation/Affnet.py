from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from architectures import AffNetFast

class Affnet:
    def __init__(self):
        PS = 32
        USE_CUDA = False

        model = AffNetFast(PS=PS)
        weightd_fname = '../../pretrained/AffNet.pth'

        checkpoint = torch.load(weightd_fname)
        model.load_state_dict(checkpoint['state_dict'])

        model.eval()
        if USE_CUDA:
            model.cuda()

