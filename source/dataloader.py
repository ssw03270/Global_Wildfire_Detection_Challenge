import torch
from torch.utils.data import Dataset
import numpy as np
import os
import rasterio

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

class ImageDatasets(Dataset):
    def __init__(self, data_type='train'):
        """
        Initializes an instance of the ImageDatasets class.

        Parameters:
        - data_type (str): Specifies the type of the data ('train', 'test', 'val'), which determines the folder from which data is loaded.
        """

        super(ImageDatasets, self).__init__()

        self.data_type = data_type

        self.input_folder_path = f'../dataset/train_img/'
        self.output_folder_path = f'../dataset/train_mask/'
        self.input_folder_path = f'/local_datasets/wildfire/train_img/'
        self.output_folder_path = f'/local_datasets/wildfire/train_mask/'

        self.input_files = [f for f in os.listdir(self.input_folder_path) if f.endswith('.tif')]
        self.output_files = [f for f in os.listdir(self.output_folder_path) if f.endswith('.tif')]

        self.input_files.sort()
        self.output_files.sort()

        self.data_length = len(self.input_files)
        self.val_length = 0.8 * self.data_length

        if data_type == 'train':
            self.input_files = self.input_files[:int(self.val_length)]
            self.output_files = self.output_files[:int(self.val_length)]
        elif data_type == 'val':
            self.input_files = self.input_files[int(self.val_length):]
            self.output_files = self.output_files[int(self.val_length):]

        self.data_length = len(self.input_files)

        print(self.data_length)

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset at the specified index.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        - dict: A dictionary containing the padded matrices and padding masks for boundary adjacency matrix, building adjacency matrix, and boundary-building adjacency matrix, as well as the boundary positions and the number of boundaries and buildings. For test data, it also returns the filename of the loaded pickle file.
        """

        load_path = self.input_folder_path + '/' + self.input_files[idx]
        with open(load_path, 'rb') as f:
            img = rasterio.open(load_path).read().transpose((1, 2, 0))
            input_img = np.float32(img)/MAX_PIXEL_VALUE

        load_path = self.output_folder_path + '/' + self.output_files[idx]
        with open(load_path, 'rb') as f:
            img = rasterio.open(load_path).read().transpose((1, 2, 0))
            output_img = np.float32(img) / MAX_PIXEL_VALUE

        return {'input_img': input_img, 'output_img': output_img}


    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
        - int: The total number of items in the dataset.
        """

        return self.data_length