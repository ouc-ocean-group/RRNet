import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import re
import numpy as np
import torch


class DronesDET(Dataset):
    def __init__(self, root_dir, transforms=None, split='train'):
        '''
        :param root_dir: root of annotations and image dirs
        :param transform: Optional transform to be applied
                on a sample.
        '''
        # get the csv
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.annotations_dir = os.path.join(root_dir, split, 'annotations')
        print('create index....')
        mdf = os.listdir(self.images_dir)
        restr = r'\w+?(?=(.jpg))'
        for index, mm in enumerate(mdf):
            mdf[index] = re.match(restr, mm).group()
        print('index created')
        self.mdf = mdf
        self.transforms = transforms

    def __len__(self):
        return len(self.mdf)

    def __getitem__(self, item):
        img_name = os.path.join(self.images_dir, '{}.jpg'.format(self.mdf[item]))
        txt_name = os.path.join(self.annotations_dir, '{}.txt'.format(self.mdf[item]))
        '''read image
        '''
        image = Image.open(img_name)

        '''read annotation
        '''
        annotation = pd.read_csv(txt_name, header=None)
        annotation = np.array(annotation).tolist()

        sample = (image, annotation)
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    def collate_fn(batch):
        imgs, annos = [], torch.zeros(len(batch), 100, 8)
        max_n = 0
        for i, batch_data in enumerate(batch):
            imgs.append(batch_data[0].unsqueeze(0))
            annos[i, :batch_data[1].size(0), :] = batch_data[1]
            max_n = max(max_n, batch_data[1].size(0))
        imgs = torch.cat(imgs)
        annos = annos[:, :max_n, :]
        return imgs, annos
