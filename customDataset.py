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



class CatdogDataset(Dataset):
    def __init__(self,data_path,target_path, filenames, transform=None):
        self.data_path=data_path
        self.target_path=target_path
        self.transform=transform
        self.filenames=filenames
        
    def __len__(self):
        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        
        input_path=self.data_path+ self.filenames[idx]
        target_path=self.target_path+ self.filenames[idx]
        
        data= read_image(input_path).to(torch.float)
        target= read_image(target_path).to(torch.float)
        
        return data, target
        