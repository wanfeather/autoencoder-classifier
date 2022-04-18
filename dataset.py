import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import pandas as pd
import os
from numpy.random import binomial

class Ladle(Dataset):

    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file, header = None)
        self.root_dir = root_dir
        self.mean = (0.1819, 0.3701, 0.6538)
        self.std = (0.2697, 0.3300, 0.2999)
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        tf = self.transform()
        x = tf(image)
        x_a = self.gaussian_noise(x) if binomial(1, .5, 1) else x

        return x_a, x

    def transform(self):
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        return tf
    
    def gaussian_noise(self, x):
        
        return x + torch.randn_like(x) * 0.2

class Ladle_labeled(Dataset):

    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file, header = None)
        self.root_dir = root_dir
        self.mean = (0.1819, 0.3701, 0.6538)
        self.std = (0.2697, 0.3300, 0.2999)
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        tf = self.transform()
        x = tf(image)
        y = torch.tensor(int(self.annotations.iloc[index, 1]))

        return x, y

    def transform(self):
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        return tf
