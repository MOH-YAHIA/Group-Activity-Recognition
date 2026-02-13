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
from models.b3_player_classifier import B3_Player_Classifier
from models.b5_player_classifier_temporal import B5_Player_Classifier_Temporal
from models.b7 import B7

setup_logger()
logger = logging.getLogger(__name__)
def evaluate(model,criterion,loader,device,pred_need,n_classes=-33):
    '''
    pred_need (bool): return labels and pred
    '''
    all_pred=[]
    all_labels=[]
    loss_sum=0
    model.eval()
    with torch.no_grad():
        for imgs,categories,labels in loader:
            imgs,categories,labels=imgs.to(device),categories.to(device),labels.to(device)
            output=model(imgs)
            loss=criterion(output,labels)
            loss_sum+=loss.item()
            _,index=output.max(dim=1)

            all_pred.extend(index.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_pred = np.array(all_pred)
    all_labels = np.array(all_labels)

    
    accurecy = np.mean(all_pred==all_labels) *100
    loss_avg = loss_sum / len(loader)
    f1Score =  f1_score(all_labels,all_pred,average='weighted')

    if not pred_need:
        return accurecy,loss_avg,f1Score
    
    return accurecy,loss_avg,f1Score,all_labels,all_pred

   

with open('config/b7.yaml','r') as file:
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
num_player_actions = conf_dict['model']['num_player_actions']
# DataLoaders
#num_workers=4,pin_memory=True
train_dataset=VolleyballPersonDataset(videos_root,annot_root,train_ids,one_frame=True)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory)

val_dataset=VolleyballPersonDataset(videos_root,annot_root,val_ids,one_frame=True)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

test_dataset=VolleyballPersonDataset(videos_root,annot_root,test_ids,one_frame=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone_inner=B3_Player_Classifier(num_player_actions)

backbone_outer=B5_Player_Classifier_Temporal(backbone_inner,num_player_actions)
backbone_outer.load_state_dict(torch.load('checkpoints/b5_player_classifier_temporal_best_model_checkpoint_sample_test.pth',map_location=device,weights_only=True)['model_state_dict'])

model=B7(backbone_outer,num_group_actions)
model=model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=conf_dict['scheduler']['mode'], factor=conf_dict['scheduler']['factor'], patience=conf_dict['scheduler']['patience'])

#  Kaggle use 2 GPU   
if torch.cuda.device_count() > 1:
    logger.info("🚀 Using %d GPUs!",torch.cuda.device_count())
    # This is the "Magic" line for T4 x2
    model = nn.DataParallel(model)

os.makedirs('checkpoints',exist_ok=True)
checkpoint={} # best checkpoint
best_loss=float('inf') 
no_update=0
def update_checkpint(epoch):
    global checkpoint
    checkpoint = {
    'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch,
    'best_loss': best_loss
    }
    #torch.save(checkpoint,f'checkpoints/b7_best_model_checkpoint_sample_test.pth')

# Train
for epoch in range(n_epoch):
    loss_sum_train=0
    all_pred=[]
    all_labels=[]
    model.train()
    model.backbone.eval()
    model.lstm1.eval()
    for ind,(imgs,categories,labels) in enumerate(train_loader):
        imgs,categories,labels=imgs.to(device),categories.to(device),labels.to(device)
        #b,f,12,3,224,224, b,f,12, b*1
        output=model(imgs) #b,8
        loss=criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_sum_train+=loss.item()
        _,index=output.max(dim=1)

        all_pred.extend(index.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        if ind%10==0:
            logger.info(f'for epoch {epoch+1} in step {ind}/{len(train_loader)}, loss: {loss.item()}')

    all_pred = np.array(all_pred)
    all_labels = np.array(all_labels)
    accurecy_train = np.mean(all_pred==all_labels) *100
    loss_avg_train = loss_sum_train/len(train_loader)
    f1Score_train =  f1_score(all_labels,all_pred,average='weighted')

    # set pred_need to false to not return labels ,pred
    accurecy_val,loss_avg_val,f1Score_val = evaluate(model,criterion,val_loader,device,False)
    
    scheduler.step(loss_avg_val) # step based on avg loss in valdiation data

    logger.info(f"\nepoch {epoch+1}/{n_epoch}")
    logger.info("Train Resault")
    logger.info(f'accurecy ->{accurecy_train}')
    logger.info(f'loss_avg ->{loss_avg_train}')
    logger.info(f'f1-score ->{f1Score_train}')
    logger.info('==========================================')
    logger.info("Validation Resault")
    logger.info(f'accurecy ->{accurecy_val}')
    logger.info(f'loss_avg ->{loss_avg_val}')
    logger.info(f'f1-score ->{f1Score_val}\n')

    if loss_avg_val < best_loss:
        update_checkpint(epoch+1)
        best_loss = loss_avg_val
        no_update = 0
        logger.info(f"New Best Model found at epoch {epoch+1}\n")
    else:
        no_update+=1
    
    if no_update>2:
        logger.info(f"Early stopping triggered at epoch {epoch+1}")
        break

logger.info(f"The best model at epoch {checkpoint['epoch']}")





# Test
logger.info(f"\n--- Test Results ---")
# set pred_need = true to get labels,pred
accurecy_test,loss_avg_test,f1Score_test,all_labels,all_pred = evaluate(model,criterion,test_loader,device,True)
logger.info('==========================================')
logger.info(f'accurecy ->{accurecy_test}')
logger.info(f'loss_avg ->{loss_avg_test}')
logger.info(f'f1-score ->{f1Score_test}\n')
        