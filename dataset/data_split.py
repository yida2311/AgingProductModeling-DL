import os
import csv
import random
import pandas as pd 


def trainval_split(root, save_dir):
    content = []
    num = [0, 0, 0, 0, 0]
    for img in os.listdir(root):
        subset = random.randint(0, 4)
        content.append(dict(img=img, subset=subset))
        num[subset] += 1
    
    df = pd.DataFrame(content, columns=['img', 'subset'])
    df.to_csv(os.path.join(save_dir, 'trainval.csv'), index=False)
    print('Number of images for each suset:  ', num)
    print('Finished!')


if __name__ == '__main__':
    root = "/remote-home/ldy/data/controller/controller-3.8"
    save_dir = "/remote-home/ldy/data/controller/"
    trainval_split(root, save_dir)