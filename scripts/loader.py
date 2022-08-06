import os
import glob
import struct
import datetime
import numpy as np
import torch
import argparse
from torch.utils.data import TensorDataset, Dataset, DataLoader

image_size = 424*240*3


class drivingData(TensorDataset):

    def __init__(self, directory):
        #go through directory and parse every .npz file
        #directory is absolute path
        self.directory = directory
        steerarray = np.array([])
        #throttle=np.array([])
        self.data_lengths = []
        self.files_list = []
        self.files_liststeer = []
        num = 0
        for filename in os.listdir(directory):
            if ("imagearray" in filename):
            #print filename
                self.files_list.append(filename)
                filedata = np.load(os.path.join(
                    directory, filename), allow_pickle=True)
                self.data_lengths.append(filedata.shape[0])
            if ("twistarray" in filename):
                filedata = np.load(os.path.join(
                    directory, filename), allow_pickle=True)
                self.files_liststeer.append(filename)
                steerarray = np.append(steerarray, filedata)
            #print "data_lengths: ", self.data_lengths

        self.labels = torch.from_numpy(steerarray).float()
        self.steer_mean = self.labels.mean()
        self.steer_std = self.labels.std()
        #print "steer mean: ", self.labels.mean()
        #print "steer stdev: ", self.labels.std()
        #normalize steer
        #self.labels = (self.labels-self.steer_mean)/self.steer_std

        #array for sample index searching
        self.start_idx = [sum(self.data_lengths[0:i])
                          for i in range(0, len(self.data_lengths))]
        #print "self.start_idx: ", self.start_idx
        #print "completed parsing dataset"
        #print "end self.data_lengths: ", len(self.data_lengths)

    def __len__(self):
        return sum(self.data_lengths)

    def __getitem__(self, idx):
        fileidx = np.searchsorted(self.start_idx, idx, side='right')-1
        #print "fileidx: ",fileidx
        local_idx = idx-self.start_idx[fileidx]
        #print "local_idx: ",local_idx
        imgdata = np.load(os.path.join(
            self.directory, self.files_list[fileidx]), allow_pickle=True)
        steerdata = np.load(os.path.join(
            self.directory, self.files_liststeer[fileidx]), allow_pickle=True)
        #extract single image/steer at the index

        image = imgdata[local_idx]
        steer = steerdata
        #convert this to tensor
        image_tensor = torch.from_numpy(image)
        image_tensor = torch.reshape(image_tensor, (3, 240, 320)).float()
        image_mean = image_tensor.mean()
        image_std = image_tensor.std()
        steer_tensor = torch.from_numpy(steer).float()
        steer_tensor = torch.unsqueeze(steer_tensor, 1)
        STR = steer_tensor[local_idx]
        #normalize image
        image_tensor = (image_tensor - image_mean)/image_std
        #return sample
        sample = (image_tensor, STR)
        #print sample
        return sample
