from data.common import list_pictures

from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader

class dukeMTMC(dataset.Dataset):
    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader

        data_path = args.datadir
        if dtype == 'train':
            data_path += '/bounding_box_train'
        elif dtype == 'test':
            data_path += '/bounding_box_test'
        else:
            data_path += '/query'

        
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

    @staticmethod
    def id(file_path):

     
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
     
        
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):

        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):

        return sorted(set(self.ids))

    @property
    def cameras(self):
 
        return [self.camera(path) for path in self.imgs]
