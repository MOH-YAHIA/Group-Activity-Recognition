import random
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from utils.image_level_dataset import VolleyballImageDataset
from models.b1 import B1
from scripts.train import train
from scripts.eval import evaluate
from scripts.final_report import Final_Report

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

with open('config/b1.yaml','r') as file:
    conf_dict = yaml.safe_load(file)

# Seed
seed=conf_dict['system']['seed']
seed_everything(seed)

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
#num_workers=4,pin_memory=True
train_dataset=VolleyballImageDataset(videos_root,annot_root,train_ids)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory)

val_dataset=VolleyballImageDataset(videos_root,annot_root,val_ids)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

test_dataset=VolleyballImageDataset(videos_root,annot_root,test_ids)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)



# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=B1(num_group_actions)
model=model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=conf_dict['scheduler']['mode'], factor=conf_dict['scheduler']['factor'], patience=conf_dict['scheduler']['patience'])

#  Kaggle use 2 GPU   
if torch.cuda.device_count() > 1:
    print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
    # This is the "Magic" line for T4 x2
    model = nn.DataParallel(model)


# Train
train('b1',model,criterion,optimizer,scheduler,train_loader,val_loader,n_epoch,device)



# Test
print(f"\n--- Test Results ---")
# set pred_need = true to get labels,pred
accurecy_test,loss_avg_test,f1Score_test,all_labels,all_pred = evaluate(model,criterion,test_loader,device,True)
print('==========================================')
print(f'accurecy ->{accurecy_test}')
print(f'loss_avg ->{loss_avg_test}')
print(f'f1-score ->{f1Score_test}\n')
        
final_report = Final_Report('b1',all_labels,all_pred)
print("Create Report in 'outputs/B1'")
final_report.creat_report()
print("Create confusion_matrix in 'outputs/B1'")
final_report.create_confusion_matrix()