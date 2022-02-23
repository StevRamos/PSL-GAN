#Taken from https://github.com/DegardinBruno/Kinetic-GAN

# Standard library imports
import os
import sys
import random
import pickle
import time

# Third party imports
import torch
import h5py
import numpy as np




class PSLDataset(torch.utils.data.Dataset):
    """Peruvian Sign Language Dataset
        Dataset: https://github.com/gissemari/PeruvianSignLanguage
    Arguments:
        data_path: the path to ".h5" data. Extracted by running create_dataset.py
    """
    def __init__(self,
                data_path,
                norm=True
                ):
        self.data_path = data_path
        self.norm = norm
        self.load_data()


    def load_data(self):
        self.data_info = h5py.File(self.data_path, 'r')
        self.data = self.data_info['data'][...]
        self.label = self.data_info['labels'][...]
        self.sample_name = self.data_info['name_labels'][...]

        self.max, self.min = self.data.max(), self.data.min()

        self.N, self.C, self.T, self.V = self.data.shape

        self.n_classes = len(np.unique(self.label))

    
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        data_numpy = 2 * ((data_numpy-self.min)/(self.max - self.min)) - 1 if self.norm else data_numpy
        label = self.label[index]
        
        return data_numpy, label