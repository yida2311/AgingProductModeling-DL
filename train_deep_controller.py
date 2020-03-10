import os
import time
import numpy as np 
import torch
import torch.nn as nn 
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset.deep_controller import DeepController, get_dataset
from utils.loss import BCELoss 
from utils.solver import LR_Scheduler, get_optimizer
from utils.metric import AverageMeter, ClsScorer
from helper import create_model_load_weights, Trainer, Evaluator, collate
from option import Options

args = Options().parse()
n_class = args.n_class

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

data_path = args.data_path
meta_path = args.meta_path

model_path = args.model_path
if not os.path.isdir(model_path): 
    os.makedirs(model_path)

log_path = args.log_path
if not os.path.isdir(log_path): 
    os.makedirs(log_path)

task_name = args.task_name
print(task_name)
###################################

evaluation = args.evaluation
test = evaluation and False
print("evaluation:", evaluation, "test:", test)

###################################
print("preparing datasets and dataloaders......")
batch_size = args.batch_size

data_time = AverageMeter("DataTime", ':6.3f')
batch_time = AverageMeter("BatchTime", ':6.3f')

dataset_train = get_dataset(data_path, meta_path, train=True)
dataloader_train = DataLoader(dataset_train, num_workers=4, batch_size=batch_size, collate_fn=collate, shuffle=True, pin_memory=True)
dataset_val = get_dataset(data_path, meta_path, train=False)
dataloader_val = DataLoader(dataset_val, num_workers=4, batch_size=batch_size, collate_fn=collate, shuffle=False, pin_memory=True)

###################################
print("creating models......")

path = os.path.join(model_path, args.path_test) if args.path_test else args.path_test
model = create_model_load_weights(n_class, evaluation, path=path)

###################################
num_epochs = args.epoch
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weight_decay
opt_args = dict(lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

optimizer = get_optimizer(model, **opt_args)
scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloader_train))
##################################

criterion = BCELoss()
if not evaluation:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')

trainer = Trainer(criterion, optimizer, n_class)
evaluator = Evaluator(n_class, test)

best_pred = 0.0
print("start training......")

for epoch in range(num_epochs):
    optimizer.zero_grad()
    tbar = tqdm(dataloader_train)
    train_loss = 0
    start_time = time.time()
    for i_batch, sample_batched in enumerate(tbar):
        print(i_batch)
        data_time.update(time.time()-start_time)
        if evaluation:  # evaluation pattern: no training
            break
        scheduler(optimizer, i_batch, epoch, best_pred)
        loss = trainer.train(sample_batched, model)
        train_loss += loss.item()

        score_train = trainer.get_scores()
        precision = score_train['precision']
        mAP = score_train['mAP']


        batch_time.update(time.time()-start_time)
        start_time = time.time()
        tbar.set_description('Train loss: %.3f; precision: %.3f; mAP: %.3f; data time: %.3f; batch time: %.3f'% 
                (train_loss / (i_batch + 1), precision, mAP, data_time.avg, batch_time.avg))
    writer.add_scalar('loss', train_loss/len(tbar), epoch)
    writer.add_scalar('mAP/train', mAP, epoch)
    writer.add_scalar('precision/train', precision, epoch)
    writer.add_scalar('distance/train', score_train['distance'], epoch)

    trainer.reset_metrics()
    data_time.reset()
    batch_time.reset()

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("evaluating...")

            if test: 
                tbar = tqdm(dataloader_test)
            else: 
                tbar = tqdm(dataloader_val)
            
            start_time = time.time()
            for i_batch, sample_batched in enumerate(tbar):
                data_time.update(time.time()-start_time)
                preds = evaluator.eval(sample_batched, model)
                score_val = evaluator.get_scores()
                batch_time.update(time.time()-start_time)
                
                tbar.set_description('precision: %.3f; mAP: %.3f; data time: %.3f; batch time: %.3f' % 
                        (score_val['precision'], score_val['mAP'], data_time.avg, batch_time.avg))
                start_time = time.time()
            
            data_time.reset()
            batch_time.reset()
            # if not (test or evaluation): 
            #     torch.save(model.state_dict(), "./saved_models/" + task_name + "-epoch" + str(epoch) + ".pth")
            
            if test:  # one epoch
                break
            else: # val log results
                score_val = evaluator.get_scores()
                evaluator.reset_metrics()
                if score_val['mAP'] > best_pred: 
                    best_pred = score_val['mAP']
                    torch.save(model.state_dict(), "./results/saved_models/" + task_name + ".pth")
            
                log = ""
                log = log + 'epoch [{}/{}] Percision: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, score_train['precision'], score_val['precision']) + "\n"
                log = log + 'epoch [{}/{}] mAP: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, score_train['mAP'], score_val['mAP']) + "\n"
                log = log + 'epoch [{}/{}] Distance: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, score_train['distance'], score_val['distance']) + "\n"
                log = log + "train: " + str(score_train) + "\n"
                log = log + "val:" + str(score_val) + "\n"
                log += "================================\n"
                print(log)
                if evaluation: 
                    break  # one peoch

                f_log.write(log)
                f_log.flush()
                writer.add_scalar('mAP/val', score_val['mAP'], epoch)
                writer.add_scalar('precision/val', score_val['precision'], epoch)
                writer.add_scalar('distance/val', score_val['distance'], epoch)

if not evaluation: 
    f_log.close()
        


