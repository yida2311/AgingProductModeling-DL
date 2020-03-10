from PIL import Image
from torchvision import transforms
import random
import torch


class Transformer(object):
    def __init__(self):
        self.crop = transforms.RandomResizedCrop(size=800, scale=(0.85, 1.0), ratio=(0.75, 1.0))
        self.rotate = transforms.RandomRotation(degrees=90, resample=Image.BILINEAR)
        self.hflip = transforms.RandomHorizontalFlip()
        self.vflip = transforms.RandomVerticalFlip()
        self.to_tensor = transforms.ToTensor()
        self.color_jitter =  transforms.ColorJitter(brightness=64.0/255, contrast=0.25, saturation=0.25, hue=0.04)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, img, target):
        if random.random() < 0.5:
            img = self.crop(img)
        img = self.rotate(img)
        if random.random() < 0.5:
            img = self.hflip(img)
        else:
            img = self.vflip(img)
        img = self.color_jitter(img)
        img = self.to_tensor(img)
        img = self.normalize(img)
        target = torch.Tensor(target)
        return img, target 


class ValTransformer(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, img, target):
        img = self.to_tensor(img)
        img = self.normalize(img)
        target = torch.Tensor(target)
        return img, target

