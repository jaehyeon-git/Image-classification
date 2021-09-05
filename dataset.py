import torch
import numpy as np
import pandas as pd

from PIL import Image, ImageTransform
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import *

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

# file 확장자 추출
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

############################ Data Augmentation ###############################
class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

# Gaussian Noise 함수
class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    
class TrainAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            RandomChoice([ColorJitter(brightness=(0.2, 3)),
                         ColorJitter(contrast=(0.2, 3)),
                         ColorJitter(saturation=(0.2, 3)),
                         ColorJitter(hue=(-0.3, 0.3))]),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

################################# Dataset ######################################
class TrainDataset(Dataset):
    def __init__(self, train_df, mean=(0.534, 0.487, 0.459), std=(0.237, 0.243, 0.251), features=False):
        self.mean = mean
        self.std = std
        self.train_df = train_df
        self.transform = None
        self.features = features

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, index):
        assert self.transform is not None, "[train_dataset] : .set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        path = self.train_df.path.iloc[index]
        image = Image.open(path).convert('RGB')
        image_transform = self.transform(image)
        label = self.train_df.label.iloc[index]
        feature_label_dict = {'age' : self.train_df['age'].iloc[index],
                         'gender' : self.train_df['gender'].iloc[index],
                         'mask' : self.train_df['mask'].iloc[index]} 
        
        # multi label을 사용하는 경우
        if self.features:
            return image_transform, torch.tensor(feature_label_dict[self.features])
        else:
            return image_transform, torch.tensor(label)


class ValidDataset(Dataset):
    def __init__(self, valid_df, mean=(0.534, 0.487, 0.459), std=(0.237, 0.243, 0.251), features=False):
        self.mean = mean
        self.std = std
        self.valid_df = valid_df
        self.transform = None
        self.features = features

    def set_transform(self, transform):
        self.transform = transform

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    def __len__(self):
        return len(self.valid_df)

    def __getitem__(self, index):
        assert self.transform is not None, "[valid_dataset] : .set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        path = self.valid_df.path.iloc[index]
        image = Image.open(path).convert('RGB')
        image_transform = self.transform(image)
        label = self.valid_df.label.iloc[index]
        feature_label_dict = {'age' : self.valid_df['age'].iloc[index],
                         'gender' : self.valid_df['gender'].iloc[index],
                         'mask' : self.valid_df['mask'].iloc[index]}
        
        # multi label을 사용하는 경우
        if self.features:
            return image_transform, torch.tensor(feature_label_dict[self.features])
        else: 
            return image_transform, torch.tensor(label)


class TestDataset(Dataset):
    def __init__(self, image_path, resize, mean=(0.534, 0.487, 0.459), std=(0.237, 0.243, 0.251)):
        self.path = image_path
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        assert self.transform is not None, "[test_dataset] : .set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = Image.open(self.path[index]).convert('RGB')
        image = self.transform(image)
        return image
