import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import logging
from sklearn.metrics import f1_score
import pandas as pd
from utils.person_level_dataset import VolleyballPersonDataset
from utils.logger import setup_logger
from scripts.train import train
from scripts.eval import evaluate
from scripts.final_report import Final_Report
from models.b3_player_classifier import B3_Player_Classifier
from models.b5_player_classifier_temporal import B5_Player_Classifier_Temporal
from models.b8 import B8

os.makedirs('logs',exist_ok=True)
log_path='logs/b8_progress.log'
setup_logger(log_path)
logger = logging.getLogger(__name__)

with open('config/b8.yaml','r') as file:
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
lr1 = conf_dict['training']['lr1']
lr2 = conf_dict['training']['lr2']
batch_size = conf_dict['training']['batch_size']
num_group_actions = conf_dict['model']['num_group_actions']
num_player_actions = conf_dict['model']['num_player_actions']

# DataLoaders
train_dataset=VolleyballPersonDataset(videos_root,annot_root,train_ids,one_frame=True,player_label=False,train=True)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory)

val_dataset=VolleyballPersonDataset(videos_root,annot_root,val_ids,one_frame=True,player_label=False,train=False)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

test_dataset=VolleyballPersonDataset(videos_root,annot_root,test_ids,one_frame=True,player_label=False,train=False)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone_inner=B3_Player_Classifier(num_player_actions)

backbone_outer=B5_Player_Classifier_Temporal(backbone_inner,num_player_actions)
backbone_outer.load_state_dict(torch.load('checkpoints/b3_player_classifier_temporal_best_model_checkpoint.pth',map_location=device,weights_only=True)['model_state_dict'])

model=B8(backbone_outer,num_group_actions)
model=model.to(device)
criterion = nn.CrossEntropyLoss()

#expert_params =list(model.lstm1.parameters())
new_params = list(model.lstm2.parameters()) + list(model.res_vis.parameters()) +list(model.classifier.parameters())
optimizer = torch.optim.AdamW([
  #  {'params': filter(lambda p: p.requires_grad, expert_params), 'lr': lr1}, 
    {'params': new_params, 'lr': lr2} 
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=conf_dict['scheduler']['mode'], factor=conf_dict['scheduler']['factor'], patience=conf_dict['scheduler']['patience'])

#  Kaggle use 2 GPU   
if torch.cuda.device_count() > 1:
    logger.debug("Using %d GPUs!",torch.cuda.device_count())
    # This is the "Magic" line for T4 x2
    model = nn.DataParallel(model)
    

# Train
os.makedirs('checkpoints',exist_ok=True)
checkpoint_path='checkpoints/b8_best_model_checkpoint.pth'
train(model,criterion,optimizer,scheduler,train_loader,val_loader,n_epoch,device,checkpoint_path,50,2,9)


# Test
logger.info(f"Test")
# set pred_need = true to get labels,pred
accurecy_test,loss_avg_test,f1Score_test,all_labels,all_pred = evaluate(model,criterion,test_loader,device,True)
logger.info(f'Loss : {loss_avg_test:.4f}')
logger.info(f'ACC  : {accurecy_test:.2f} %')
logger.info(f'F1   : {f1Score_test:.2f} %\n')

output_path='outputs/B8'      
os.makedirs(output_path,exist_ok=True)
final_report = Final_Report(output_path,all_labels,all_pred,for_group=True)
logger.info(f"Create Report in {output_path}")
final_report.creat_report()
logger.info(f"Create Confusion Matrix in {output_path}")
final_report.create_confusion_matrix()
        
        