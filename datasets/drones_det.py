import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import re
import numpy as np
import torch
from datasets.transforms import *


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
        mdf = os.listdir(self.images_dir)
        restr = r'\w+?(?=(.jpg))'
        for index, mm in enumerate(mdf):
            mdf[index] = re.match(restr, mm).group()
        self.mdf = mdf
        self.transforms = transforms

    def __len__(self):
        return len(self.mdf)

    def __getitem__(self, item):
        # name = '0000204_01028_d_0000194'
        name = self.mdf[item]
        # name = self.mdf[100]

        img_name = os.path.join(self.images_dir, '{}.jpg'.format(name))
        txt_name = os.path.join(self.annotations_dir, '{}.txt'.format(name))
        '''read image
        '''
        image = Image.open(img_name)

        '''read annotation
        '''
        annotation = pd.read_csv(txt_name, header=None)
        annotation = np.array(annotation)
        annotation = annotation[annotation[:, 5] != 11]
        sample = (image, annotation)

        if self.transforms:
            sample = self.transforms(sample)
        return sample[0], sample[1], name

    @staticmethod
    def collate_fn(batch):
        max_n = 0
        for i, batch_data in enumerate(batch):
            max_n = max(max_n, batch_data[1].size(0))
        imgs, annos, names = [], torch.zeros(len(batch), max_n, 8), []
        for i, batch_data in enumerate(batch):
            imgs.append(batch_data[0].unsqueeze(0))
            annos[i, :batch_data[1].size(0), :] = batch_data[1][:, :8]
            names.append(batch_data[2])
        imgs = torch.cat(imgs)
        return imgs, annos, names


    @staticmethod
    def collate_fn_centernet(batch):
        max_n = 0
        for i, batch_data in enumerate(batch):
            max_n = max(max_n, batch_data[1].size(0))
        imgs = []
        hms = []
        whs = []
        regs = []
        inds = []
        reg_masks = []
        annos, names = torch.zeros(len(batch), max_n, 8), []
        trans = TransToHM_Origin()
        # trans = TransToHM()
        for i, batch_data in enumerate(batch):
            imgs.append(batch_data[0].unsqueeze(0))
            data_n = batch_data[1].size(0)
            annos[i, :batch_data[1].size(0), :] = batch_data[1][:, :8]
            img, hm, wh, ind, reg, reg_mask = trans(batch_data, max_n, data_n)
            hms.append(hm)
            whs.append(wh)
            inds.append(ind)
            regs.append(reg)
            reg_masks.append(reg_mask)
            names.append(batch_data[2])

        imgs = torch.cat(imgs)
        hms = torch.stack(hms)
        whs = torch.stack(whs)
        inds = torch.stack(inds)
        regs = torch.stack(regs)
        reg_masks = torch.stack(reg_masks)
        # annos = hms, whs, regs, inds, reg_masks
        return imgs, hms, whs, regs, inds, reg_masks, annos, names



