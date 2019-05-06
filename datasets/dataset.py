import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np


class ICCVDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        '''
        :param root_dir: root of annotations and image dirs
        :param transform: Optional transform to be applied
                on a sample.
        '''
        # get the csv
        self.images_dir = os.path.join(root_dir, 'images')
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        print('create index....')
        mdf = os.listdir(self.images_dir)
        restr = r'\w+?(?=(.jpg))'
        for index, mm in enumerate(mdf):
            mdf[index] = re.match(restr, mm).group()
        print('index created')
        self.mdf = mdf
        self.transform = transform

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
        if self.transform:
            sample = self.transform(sample)
        return sample
