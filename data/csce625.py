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
class csce625(dataset.Dataset):
    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader

        data_path = args.datadir
        label_path = args.datadir
        if dtype == 'test':
            data_path += '/gallery'
            label_path += '/galleryInfo.txt'
        else:
            data_path += '/query'
            label_path += '/queryInfo.txt'

        
        self.labels = self.build_label_dict(label_path)

        self.imgs = [path for path in list_pictures(data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    # Generates dictionary from paths and labels in Info.txt files
    def build_label_dict(self, labels_path):
        if not labels_path:
            return

        labels = {}
        with open(labels_path) as file:
            for line in file:
                (filename, label) = line.split('\t')
                # filename = path.split('/')[-1].split('.')[0]
                labels[filename] = label.strip('\n')
        return labels

    def id(self, file_path):
        """
        Matches filepath to corresponding label from label dict
        :param file_path: unix style file path
        :return: person id
        """
        filename, _ = path.splitext(path.basename(file_path))
        return int(self.labels[str(filename)])

    def filename(self, file_path):
        filename, _ = path.splitext(path.basename(file_path))
        return filename

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]
    
    @property
    def names(self):
        """
        :return: filenames list corresponding to data image paths
        """
        return [self.filename(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))
