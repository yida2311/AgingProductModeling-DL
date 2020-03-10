import os
import random
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
import pandas as pd 

from .transformer import Transformer, ValTransformer


def pil_loader(path):
    img = Image.open(path).convert('RGB')
    return img


class DeepController(data.Dataset):
    def __init__(self, root, meta, train=True, transforms=None):
        super(DeepController, self).__init__()
        self.root = root
        self.train = train
        self.transforms = transforms

        samples = []
        df = pd.read_csv(meta)
        if self.train:
            for i in df.index:
                cnt = df.iloc[i]
                if int(cnt[1]) != 4:
                    img = cnt[0]
                    target = self._parse_name2target(img)
                    samples.append((img, target))
        else:
            for i in df.index:
                cnt = df.iloc[i]
                if int(cnt[1]) == 4:
                    img = cnt[0]
                    target = self._parse_name2target(img)
                    samples.append((img, target))
        self.samples = samples
        random.shuffle(self.samples)
    
    def __getitem__(self, index):
        name, target = self.samples[index]
        img = pil_loader(os.path.join(self.root, name))
        if self.transforms:
            img, target = self.transforms(img, target)

        return dict(img=img, label=target)
    
    def __len__(self):
        return len(self.samples)

    def _parse_name2target(self, name):
        digit = name.split('.')[0]
        if "(" in digit:
            digit = digit.split(' ')[0]
        # print(digit)
        target = [int(c) for c in digit]
        assert len(target) == 12, 'wrong name {}'.format(name) + str(len(target))
        return np.array(target, dtype=np.bool)


def get_dataset(root, meta, train=True):
    if train:
        return DeepController(root, meta, transforms=Transformer())
    else:
        return DeepController(root, meta, train=False, transforms=ValTransformer())