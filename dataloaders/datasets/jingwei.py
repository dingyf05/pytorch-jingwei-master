from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class JingweiSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 4

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('jingwei'),
                 split='train',
                 ):
        """
        :param base_dir: path to Jingwei dataset directory
        :param transform: transform to apply
        """
        super().__init__()
        self.split = split
        self.args = args

        # 读取预存的.npy数据，请修改读取的路径
        self.unit_size = args.base_size
        save_dir = os.path.join('/data/dingyifeng/pytorch-deeplab-xception-master/npy_process', str(self.unit_size))

        if split == "train":
            self.images = np.load(os.path.join(save_dir, 'train_img.npy'))
            self.categories = np.load(os.path.join(save_dir, 'train_label.npy'))
        elif split == 'val':
            self.images = np.load(os.path.join(save_dir, 'val_img.npy'))
            self.categories = np.load(os.path.join(save_dir, 'val_label.npy'))

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img = Image.fromarray(self.images[index])
        _target = Image.fromarray(self.categories[index])
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # tr.RandomGaussianBlur(),
            # tr.FixedResize(self.args.crop_size),
            # tr.RandomCrop(self.args.crop_size),
            # tr.RandomCutout(n_holes=1, cut_size=128),
            # tr.RandomRotate(30),
            tr.RandomRotate_v2(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'Jingwei2019(split=' + str(self.split) + ')'



