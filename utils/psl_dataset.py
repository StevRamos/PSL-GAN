#Taken from https://github.com/DegardinBruno/Kinetic-GAN

# Standard library imports
import os
import sys
import random
import time

# Third party imports
import torch
import h5py
import pickle
import numpy as np
from sklearn import preprocessing


class PSLDataset(torch.utils.data.Dataset):
    """Peruvian Sign Language Dataset
        Dataset: https://github.com/gissemari/PeruvianSignLanguage
    Arguments:
        data_path: the path to ".pk" data. Extracted by running create_dataset.py
    """
    def __init__(self,
                data_path,
                norm=True,
                classes=None
                ):
        self.data_path = data_path
        self.norm = norm
        self.classes = classes
        self.load_data()

    def load_data(self):
        self.data_info = pickle.load(open(self.data_path, 'rb'))
        
        self.data = self.data_info['data'] #[...]
        self.sample_name = self.data_info['name_labels'] #[...]

        
        if (self.classes is not None) and (len(self.classes)>0):
            print(f"it will be used the following classes")
            print(f"{self.classes=}")
            condition = np.isin(np.array(self.sample_name), self.classes)
            self.data = self.data[condition]
            self.sample_name = np.array(self.sample_name)[condition]

            assert sorted(np.unique(self.sample_name)) == sorted(self.classes), "Some classes were not found in the dataset"

        else: 
            self.classes = list(set(self.sample_name))
            print("All the classes will be used")
            print(f"{self.classes=}")

        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.sample_name)
        self.label = self.label_encoder.transform(self.sample_name)

        
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
        name_label = self.sample_name[index]
        
        return data_numpy, label, name_label 

    def get_data_by_class(self, class_name):
        condition = np.isin(np.array(self.sample_name), [class_name])
        data_numpy = self.data[condition]
        data_numpy = 2 * ((data_numpy-self.min)/(self.max - self.min)) - 1 if self.norm else data_numpy
        label = self.label[condition]
        name_label = np.array(self.sample_name)[condition]

        return data_numpy, label, name_label 