import torch
import os
import cv2
import yaml
import logging
from .augmentation import RandAugment
LOGGER = logging.get_logger(__name__)

class LoadImagesAndLabels(torch.utils.data.Dataset):
    
    def __init__(self, csv, data_folder, preprocess=False, augment=False):
        self.csv = csv 
        self.data_folder = data_folder 
        self.augment = augment 
        self.preprocess = preprocess
        if augment:
            self.augmenter = RandAugment(num_layers=2)
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index,):
        item = self.csv.iloc[index]
        path = os.path.join(self.data_folder, item.path)
        assert os.path.isfile(path), LOGGER.error(f'this image : {path} is corrupted')
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            LOGGER.info(f'this image : {path} is corrupted')
        label = item.label
        if self.augment:
            img = augmenter(img)
        if self.preprocess:
            img = self.preprocess(img)
        return img,label
            

