import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
import os
import pandas as pd

class Ladle_test(Dataset):

    def __init__(self, root_dir):
        self.annotations = '1.jpg'
        self.root_dir = root_dir
        self.mean = (0.2657, 0.3733, 0.5235)
        self.std = (0.2913, 0.3441, 0.3112)
    
    def __len__(self):
        return 1

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations)
        image = io.imread(img_path)
        tf = self.transform()
        tensor = tf(image)

        return tensor
    
    def transform(self):
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        return tf

class Ladle_val(Dataset):

    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mean = (0.2657, 0.3733, 0.5235)
        self.std = (0.2913, 0.3441, 0.3112)
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        tf = self.transform()
        tensor = tf(image)

        return tensor
    
    def transform(self):
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        return tf

def test(model, device, num):
    test_set = Ladle_test('crop_data')
    data_loader = DataLoader(test_set, 1, False)

    model.eval()
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            img = model(x)
            utils.save_image(img.detach(), './gen_img/{}.png'.format(num + 1))
    model.train()

def val(model, device):
    val_set = Ladle_val('data.csv', 'crop_data')
    data_loader = DataLoader(val_set, 64, False)
    num = 0

    model.eval()
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            img = model(x)
            utils.save_image(img.detach(), './validation/{}.png'.format(num))
            num += 1
