import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

from PIL import Image
from IPython.display import display
import warnings
from sklearn.preprocessing import normalize
warnings.filterwarnings('ignore')



class RedCNN(nn.Module):
    ##k=filter size, f=number of filters
    def __init__(self,k=5,f=96):
        
        super().__init__()
        ##Encoding Layers
        self.conv1=nn.Conv2d(in_channels=1,out_channels=f,kernel_size=k,padding=2,bias=True)
        self.conv2=nn.Conv2d(in_channels=f,out_channels=f,kernel_size=k,padding=2,bias=True)
        self.conv3=nn.Conv2d(in_channels=f,out_channels=f,kernel_size=k,padding=2,bias=True)
        self.conv4=nn.Conv2d(in_channels=f,out_channels=f,kernel_size=k,padding=2, bias= True)
        
        #Decoder layers
        
        self.d_conv1=nn.ConvTranspose2d(in_channels=f,out_channels=f,kernel_size=k,padding=2,bias=True)
        self.d_conv2=nn.ConvTranspose2d(in_channels=f,out_channels=f,kernel_size=k,padding=2,bias=True)
        self.d_conv3=nn.ConvTranspose2d(in_channels=f,out_channels=f,kernel_size=k,padding=2,bias=True)
        self.d_conv4=nn.ConvTranspose2d(in_channels=f,out_channels=f,kernel_size=k,padding=2,bias=True)
        self.d_output=nn.ConvTranspose2d(in_channels=f,out_channels=1,kernel_size=k,padding=2, bias=True)
        
        self.batchnorm=nn.BatchNorm2d(f)
        
        
    def forward(self,x):
        xinit=x
        x=F.relu(self.batchnorm(self.conv1(x)))
        x2=x.clone()
        x=F.relu(self.batchnorm(self.conv2(x)))
        x=F.relu(self.batchnorm(self.conv3(x)))
        x4=x.clone()
        x=F.relu(self.batchnorm(self.conv4(x)))
        
        ##dencode
        x=F.relu(self.batchnorm(self.d_conv1(x))+x4)
        x=F.relu(self.batchnorm(self.d_conv2(x)))
        x=F.relu(self.batchnorm(self.d_conv3(x))+x2)
        x=F.relu(self.batchnorm(self.d_conv4(x)))
        
        x=self.d_output(x)+xinit
        
        return x