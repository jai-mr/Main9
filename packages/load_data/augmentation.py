import abc

import cv2
import torchvision.transforms as T
import albumentations as alb
from albumentations.pytorch import ToTensor
import albumentations.pytorch.transforms as AT
import numpy as np

class AlbumentationTrans:
    
    def __init__(self, transform):
        self.album_transform = transform

    def __call__(self, img):
        img = np.array(img)
        return self.album_transform(image=img)['image']

class AugmentationBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class MNIST_Transforms(AugmentationBase):

    def build_train(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))])

    def build_test(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))])


class CIFAR10_Transforms(AugmentationBase):

    def build_train(self):
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def build_test(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])


class CIFAR10_AlbumTrans(AugmentationBase):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    def __init__(self, augs= None):
      self.augs = augs

 
    def build_train(self):

        train_trans = alb.Compose([
            alb.PadIfNeeded(min_height=36, min_width=36),
            alb.RandomCrop(height=32, width=32),
            alb.HorizontalFlip(),
            alb.Normalize(mean=self.mean,
                        std=self.std),
            alb.Cutout(num_holes=4),
            AT.ToTensor()
        ])
        return AlbumentationTrans(train_trans)

    def build_test(self):
        test_trans = alb.Compose([
            alb.Normalize(
                mean=self.mean,
                std=self.std
            ),
            AT.ToTensor()
        ]) 
        return AlbumentationTrans(test_trans)
 
