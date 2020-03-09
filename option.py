import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Multi-label Classification')
        parser.add_argument('--n_class', type=int, default=4, help='num of labels')
        parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--epoch', type=int, default=120, help='num of training epochs')
        parser.add_argument('--data_path', type=str, help='path to dataset where images store')
        parser.add_argument('--meta_path', type=str, help='path to meta_file where images name store')
        parser.add_argument('--model_path', type=str, help='path to store trained model files, no need to include task specific name')
        parser.add_argument('--log_path', type=str, help='path to store tensorboard log files, no need to include task specific name')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--evaluation', action='store_true', default=False, help='evaluation only')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--path_test', type=str, default="", help='name for test model path')

        self.parser = parser
    
    def parse(self):
        args = self.parser.parse_args()
        return args
    