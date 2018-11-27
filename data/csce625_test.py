from data.common import list_pictures

from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from os import path

import pdb

"""
CSCE 625

Processes images from validation set for CSCE 625
"""
class csce625_test(dataset.Dataset):
    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader

        data_path = args.datadir
        if dtype == 'test':
            data_path += '/gallery'
        else:
            data_path += '/query'

        self.imgs = [path for path in list_pictures(data_path)]

        #self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, ''

    def __len__(self):
        return len(self.imgs)

    def filename(self, file_path):
        filename, _ = path.splitext(path.basename(file_path))
        return filename
    
    @property
    def names(self):
        """
        :return: filenames list corresponding to data image paths
        """
        return [self.filename(path) for path in self.imgs]
