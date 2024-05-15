from torch import nn
import torch
import numpy as np

class Anchors(nn.Module):
        def __init__(self, pyramid_levels=[2,3,4,5,6],):
            super(Anchors,self).__init__();
