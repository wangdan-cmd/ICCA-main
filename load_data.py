
from scipy.io import loadmat, savemat

from util import BackgroundGenerator
import numpy as np


class MyDataset(Dataset):
    def __init__(self, image_data, text_data, image_labels, text_labels, multi_labels):
        self.image_data = image_data
        self.text_data = text_data
        self.image_labels = image_labels
        self.text_labels = text_labels
        self.multi_labels = multi_labels

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        text = self.text_data[idx]
        image_label = self.image_labels[idx]
        text_label = self.text_labels[idx]
        multi_label = self.multi_labels[idx]
        return image, text, image_label, text_label, multi_label
