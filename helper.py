import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from models.resnet import resnet34, resnet50
from utils.metric import AverageMeter, ClsScorer

torch.backends.cudnn.deterministic = True


def create_model_load_weights(n_class, evaluation=False, path=None):
    if not evaluation:
        model = resnet50(pretrained=True, num_classes=n_class)
    else:
        model = resnet50(pretrained=False, num_classes=n_class)
        model.load_state_dict(torch.load(path))
    # model = nn.DataParallel(model)
    model = model.cuda()
    return model

def collate(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [b[key] for b in batch_dict]
    return batch_dict


class Trainer(object):
    def __init__(self, criterion, optimizer, n_class):
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_class = n_class
        self.metrics = ClsScorer(n_class)
    
    def get_scores(self):
        return self.metrics.get_scores()

    def reset_metrics(self):
        self.metrics.reset()
    
    def train(self, sample, model):
        model.train()
        imgs = sample['img'].cuda()
        labels = sample['label'].cuda()

        outputs = model.forward(imgs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        preds = outputs.cpu().numpy() # N x K
        preds = (preds > 0.5)
        self.metrics.update(labels, preds)
        
        return loss


class Evaluator(object):
    def __init__(self, n_class, test=False):
        self.n_class = n_class
        self.test = test
        self.metrics = ClsScorer(n_class)
    
    def get_scores(self):
        return self.metrics.get_scores()

    def reset_metrics(self):
        self.metrics.reset()
    
    def eval(self, sample, model):
        model.val()
        with torch.no_grad():
            imgs = sample['img'].cuda()
            if not self.test:
                labels = sample['label'].cuda()
            
            outputs = model.forward(imgs)
            preds = outputs.cpu().numpy() # N x K
            preds = (preds > 0.5)
            self.metrics.update(labels, preds)
        
        return preds
