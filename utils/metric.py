import math
import numpy as np 


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ClsScorer(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.count = 0  # 总例数
        self.right = 0 # 完全准确的例数
        self.right_labels = np.zeros(n_classes)
        self.distance = 0
    
    def update(self, labels, preds):
        # N x K
        size = labels.shape[0]
        # print(labels.shape, preds.shape)
        results = np.array(labels == preds, dtype=np.int)
        # print(results.shape)
        hist = results.sum(axis=1)

        self.count += size
        self.right += np.array(hist == self.n_classes, dtype=np.int).sum()
        self.right_labels += results.sum(axis=0)
        self.distance += (self.n_classes - hist).sum()
    
    def get_scores(self):
        """Return accuracy socre evaluation result.
            - precision
            - mAP
            - APs
            - Hanming distance
        """
        precision = self.right / self.count
        APs = self.right_labels / self.count
        mAP = np.mean(APs)
        distance = self.distance / self.count

        return {'precision': precision,
                'APs': APs,
                'mAP': mAP,
                'distance': distance
                }
    
    def reset(self):
        self.count = 0 
        self.right = 0 
        self.right_labels = np.zeros(self.n_classes)
        self.distance = 0



        