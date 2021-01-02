import torch
from torch.utils.data import Dataset

from PIL import Image
from glob import glob
import os.path as osp

class miniimagenetDataset(Dataset):
    def __init__(self, data_root, transform):
        super().__init__()
        file_names = glob(osp.join(data_root, "*"))

        self.fnames = file_names
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.fnames[idx])
        img = self.transform(img)
        return img, 0 # label for aligned with torchvision dataset

    def __len__(self):
        return len(self.fnames)
