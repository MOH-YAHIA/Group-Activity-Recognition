import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import logging
from utils.image_level_dataset import VolleyballImageDataset
from utils.logger import setup_logger
from models.b1 import B1
from scripts.train import train
from scripts.eval import evaluate
from scripts.final_report import Final_Report

os.makedirs('logs',exist_ok=True)
log_path='logs/b1_progress.log'
setup_logger(log_path)
logger=logging.getLogger(__name__)

with open('config/b1.yaml','r') as file:
    conf_dict = yaml.safe_load(file)

# Dataset path
annot_root=conf_dict['paths']['annot_root']
videos_root=conf_dict['paths']['videos_root']

# Split
train_ids = conf_dict['splits']['train_ids']
val_ids = conf_dict['splits']['val_ids']
test_ids = conf_dict['splits']['test_ids']

# Hyperparameters
num_workers = conf_dict['training']['num_workers']
pin_memory = conf_dict['training']['pin_memory']
n_epoch = conf_dict['training']['n_epoch']
lr = conf_dict['training']['lr']
batch_size = conf_dict['training']['batch_size']
num_group_actions = conf_dict['model']['num_group_actions']

# DataLoaders
train_dataset=VolleyballImageDataset(videos_root,annot_root,train_ids,one_frame=True,train=True)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory)

val_dataset=VolleyballImageDataset(videos_root,annot_root,val_ids,one_frame=True,train=False)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

test_dataset=VolleyballImageDataset(videos_root,annot_root,test_ids,one_frame=True,train=False)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)



# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=B1(num_group_actions)
model=model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=conf_dict['scheduler']['mode'], factor=conf_dict['scheduler']['factor'], patience=conf_dict['scheduler']['patience'])


# Train
os.makedirs('checkpoints',exist_ok=True)
checkpoint_path='checkpoints/b1_best_mode_checkpoint.pth'
train(model,criterion,optimizer,scheduler,train_loader,val_loader,n_epoch,device,checkpoint_path,10)



# Test
logger.info(f"Test")
# set pred_need = true to get labels,pred
accurecy_test,loss_avg_test,f1Score_test,all_labels,all_pred = evaluate(model,criterion,test_loader,device,True)
logger.info(f'Loss  : {loss_avg_test:.4f}')
logger.info(f'ACC % : {accurecy_test:.4f}')
logger.info(f'F1 %  : {f1Score_test:.4f}\n')
        
os.makedirs('outputs/B1',exist_ok=True)
output_path='outputs/B1'
final_report = Final_Report(output_path,all_labels,all_pred)
logger.info(f"Create Report in {output_path}")
final_report.creat_report()
logger.info(f"Create confusion_matrix in {output_path}")
final_report.create_confusion_matrix()