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
from models.b5_player_classifier_temporal import B5_Player_Classifier_Temporal
from models.b3_player_classifier import B3_Player_Classifier
from scripts.final_report import Final_Report

os.makedirs('logs',exist_ok=True)
log_path='logs/b5_player_classifier_temporal_progress.log'
setup_logger(log_path)
logger=logging.getLogger(__name__)

with open('config/b5_player_classifier_temporal.yaml','r') as file:
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
num_player_actions = conf_dict['model']['num_player_actions']
player_labels_weight = conf_dict['training']['player_labels_weight']

#we need to set the order of weights as same as the we mapped labels
player_action_dct = {
    'waiting': 0, 'setting': 1, 'digging': 2, 
    'falling': 3,'spiking': 4, 'blocking': 5,
    'jumping': 6, 'moving': 7, 'standing': 8
}

weights = [0]*9
for label,index in player_action_dct.items():
    weights[index]=player_labels_weight[label]


# DataLoaders
train_dataset=VolleyballPersonDataset(videos_root,annot_root,train_ids,one_frame=False,player_label=True,train=True)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory)

val_dataset=VolleyballPersonDataset(videos_root,annot_root,val_ids,one_frame=False,player_label=True,train=False)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

test_dataset=VolleyballPersonDataset(videos_root,annot_root,test_ids,one_frame=False,player_label=True,train=False)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torch.tensor(weights , dtype=torch.float32).to(device)
backbone=B3_Player_Classifier(num_player_actions)
backbone.load_state_dict(torch.load('/kaggle/input/datasets/myahiia/b3-player-classifier-dataset/Group-Activity-Recognition/checkpoints/b3_player_classifier_best_model_checkpoint.pth',map_location=device,weights_only=True)['model_state_dict'])
model=B5_Player_Classifier_Temporal(backbone,num_player_actions)
model=model.to(device)
criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=conf_dict['scheduler']['mode'], factor=conf_dict['scheduler']['factor'], patience=conf_dict['scheduler']['patience'])

#  Kaggle use 2 GPU   
if torch.cuda.device_count() > 1:
    logger.debug("Using %d GPUs!",torch.cuda.device_count())
    # This is the "Magic" line for T4 x2
    model = nn.DataParallel(model)   

def evaluate(model,criterion,loader,device,pred_need):
    '''
    pred_need (bool): return labels and pred
    '''
    all_pred=[]
    all_labels=[]
    loss_sum=0
    model.eval()
    with torch.no_grad():
        for ind,(imgs,players_labels,_) in enumerate(loader):
            imgs,players_labels=imgs.to(device),players_labels.to(device)
            #b,f,12,3,w,h   b,f,12,    b,1
            output=model(imgs) #b,p,9
            output=output.reshape(-1,9) #b*12,9
            labels=players_labels[:,4,:] #middel frame label b,12
            labels=labels.reshape(-1) #b*12
            loss=criterion(output,labels)
            loss_sum+=loss.item()
            _,index=output.max(dim=1)

            all_pred.extend(index.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_pred = np.array(all_pred)
    all_labels = np.array(all_labels)

    
    accurecy = np.mean(all_pred==all_labels) *100
    loss_avg = loss_sum / len(loader)
    f1Score =  f1_score(all_labels,all_pred,average='weighted') *100

    if not pred_need:
        return accurecy,loss_avg,f1Score
    return accurecy,loss_avg,f1Score,all_labels,all_pred




os.makedirs('checkpoints',exist_ok=True)
# i made a typo here it should be b5
checkpoint_path='checkpoints/b3_player_classifier_temporal_best_model_checkpoint.pth'
best_loss=float('inf') 
no_update=0
choosen_epoch=0
def update_checkpint(epoch):
    checkpoint = {
    'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch,
    'best_loss': best_loss
    }
    torch.save(checkpoint,checkpoint_path)

for epoch in range(n_epoch):
    loss_sum_train=0
    all_pred=[]
    all_labels=[]
    model.train()
    target_model=model.module if isinstance(model, torch.nn.DataParallel) else model
    for i in range(7):
        target_model.backbone[i].eval() 
    for ind,(imgs,players_labels,_) in enumerate(train_loader):
        imgs,players_labels=imgs.to(device),players_labels.to(device)
        #b,f,12,3,w,h   b,f,12,    b,1
        output=model(imgs) #b,p,9
        output=output.reshape(-1,9) #b*12,9
        labels=players_labels[:,4,:] #middel frame label b,12
        labels=labels.reshape(-1) #b*12
        loss=criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_sum_train+=loss.item()
        _,index=output.max(dim=1)

        all_pred.extend(index.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if ind%100==0:
            logger.info(f'Epoch [{epoch+1}/{n_epoch}] | Step [{ind+1}/{len(train_loader)}] | Loss: {loss.item():0.4f}')

    all_pred = np.array(all_pred)
    all_labels = np.array(all_labels)
    loss_avg_train = loss_sum_train/len(train_loader)
    accurecy_train = np.mean(all_pred==all_labels) *100
    f1Score_train =  f1_score(all_labels,all_pred,average='weighted') *100

    logger.info("Train")
    logger.info(f'Loss : {loss_avg_train:.4f}')
    logger.info(f'ACC  : {accurecy_train:.2f} %')
    logger.info(f'F1   : {f1Score_train:.2f} %')

    logger.info("Validation")
    # set pred_need to false to not return labels,pred
    accurecy_val,loss_avg_val,f1Score_val = evaluate(model,criterion,val_loader,device,False)
    scheduler.step(loss_avg_val) # step based on avg loss in valdiation data


    logger.info(f'Loss : {loss_avg_val:.4f}')
    logger.info(f'ACC  : {accurecy_val:.2f} %')
    logger.info(f'F1   : {f1Score_val:.2f} %\n')

    if loss_avg_val < best_loss:
        update_checkpint(epoch+1)
        best_loss = loss_avg_val
        no_update = 0
        choosen_epoch = epoch+1
        logger.info(f"New Best Model found at epoch {epoch+1}\n")
    else:
        no_update+=1
    
    if no_update>2:
        logger.warning(f"Early stopping triggered at epoch {epoch+1}\n")
        break
logger.info(f"Best Model found at epoch {choosen_epoch}\n")




# Test
logger.info(f"Test")
# set pred_need = true to get labels,pred
accurecy_test,loss_avg_test,f1Score_test,all_labels,all_pred = evaluate(model,criterion,test_loader,device,True)
logger.info(f'Loss : {loss_avg_test:.4f}')
logger.info(f'ACC  : {accurecy_test:.2f} %')
logger.info(f'F1   : {f1Score_test:.2f} %\n')
        
os.makedirs('outputs/B5_Player_Classifier_Temporal',exist_ok=True)
output_path='outputs/B5_Player_Classifier_Temporal'
final_report = Final_Report(output_path,all_labels,all_pred,for_group=False)
logger.info(f"Create Report in {output_path}")
final_report.creat_report()
logger.info(f"Create Confusion Matrix in {output_path}")
final_report.create_confusion_matrix()

        
