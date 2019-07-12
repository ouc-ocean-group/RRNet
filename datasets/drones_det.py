import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import re
from datasets.transforms import *
import cv2


class DronesDET(Dataset):
    def __init__(self, root_dir, transforms=None, split='train', with_road_map=False):
        '''
        :param root_dir: root of annotations and image dirs
        :param transform: Optional transform to be applied
                on a sample.
        '''
        # get the csv
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.annotations_dir = os.path.join(root_dir, split, 'annotations')
        self.roadmap_dir = os.path.join(root_dir, split, 'roadmap')
        mdf = os.listdir(self.images_dir)
        restr = r'\w+?(?=(.jpg))'
        for index, mm in enumerate(mdf):
            mdf[index] = re.match(restr, mm).group()
        self.mdf = mdf
        self.transforms = transforms
        self.with_road_map = with_road_map

    def __len__(self):
        return len(self.mdf)

    def __getitem__(self, item):
        name = self.mdf[item]
        img_name = os.path.join(self.images_dir, '{}.jpg'.format(name))
        txt_name = os.path.join(self.annotations_dir, '{}.txt'.format(name))
        # read image
        image = Image.open(img_name).convert("RGB")

        # read annotation
        annotation = pd.read_csv(txt_name, header=None)
        annotation = np.array(annotation)[:, :8]
        annotation = annotation[annotation[:, 5] != 11]

        # read road segmentation
        roadmap = None
        if self.with_road_map:
            roadmap_name = os.path.join(self.roadmap_dir, '{}.jpg'.format(name))
            roadmap = cv2.imread(roadmap_name)

        sample = (image, annotation, roadmap)

        if self.transforms:
            sample = self.transforms(sample)
        return sample + (name,)

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
    def collate_fn_ctnet(batch):
        max_n = 0
        for i, batch_data in enumerate(batch):
            max_n = max(max_n, batch_data[1].size(0))
        imgs, hms, names = [], [], []
        batchsize = len(batch)
        annos, whs, offsets, inds, reg_masks = \
            torch.zeros(batchsize, max_n, 8), \
            torch.zeros(batchsize, max_n, 2), \
            torch.zeros(batchsize, max_n, 2), \
            torch.zeros(batchsize, max_n, 1), \
            torch.zeros(batchsize, max_n, 1)

        for i, batch_data in enumerate(batch):
            imgs.append(batch_data[0].unsqueeze(0))
            annos[i, :batch_data[1].size(0), :] = batch_data[1][:, :8]
            hms.append(batch_data[2].unsqueeze(0))
            whs[i, :batch_data[3].size(0), :] = batch_data[3]
            inds[i, :batch_data[4].size(0), :] = batch_data[4]
            offsets[i, :batch_data[5].size(0), :] = batch_data[5]
            reg_masks[i, :batch_data[6].size(0), :] = batch_data[6]
            names.append(batch_data[7])
        imgs = torch.cat(imgs)
        hms = torch.cat(hms)
        return imgs, annos, hms, whs, inds, offsets, reg_masks, names
