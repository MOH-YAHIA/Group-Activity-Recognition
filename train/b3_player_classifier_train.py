import random
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from sklearn.metrics import f1_score
import pandas as pd
from utils.person_level_dataset import VolleyballPersonDataset
from models.b3_player_classifier import B3_Player_Classifier


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model,criterion,loader,device,pred_need,n_classes):
    '''
    pred_need (bool): return labels and pred
    '''
    all_pred=[]
    all_categories=[]
    loss_sum=0
    model.eval()
    with torch.no_grad():
        for imgs,categories,labels in loader:
            imgs,categories,labels=imgs.to(device),categories.to(device),labels.to(device)
            output=model(imgs)
            output=output.reshape(-1,n_classes)
            categories=categories.reshape(-1)
            loss=criterion(output,categories)
            loss_sum+=loss.item()
            _,index=output.max(dim=1)

            all_pred.extend(index.cpu().numpy())
            all_categories.extend(categories.cpu().numpy())

    all_pred = np.array(all_pred)
    all_categories = np.array(all_categories)

    
    accurecy = np.mean(all_pred==all_categories) *100
    loss_avg = loss_sum / len(loader)
    f1Score =  f1_score(all_categories,all_categories,average='weighted')

    if not pred_need:
        return accurecy,loss_avg,f1Score
    
    return accurecy,loss_avg,f1Score,all_categories,all_pred

def train(baseline,model,criterion,optimizer,scheduler,train_loader,val_loader,n_epoch,device,n_classes):    
    logs=[] # metrics resualt for every epoch 
    checkpoint={} # best checkpoint
    best_loss=float('inf') 
    def update_checkpint(epoch):
        nonlocal checkpoint
        checkpoint = {
        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss
        }
    for epoch in range(n_epoch):
        loss_sum_train=0
        all_pred=[]
        all_categories=[]
        model.train()
        for ind,(imgs,categories,labels) in enumerate(train_loader):
            imgs,categories,labels=imgs.to(device),categories.to(device),labels.to(device)
            #b*12*3*224*224,  b*12*1,  b*1
            output=model(imgs) #b,12,c
            output=output.reshape(-1,n_classes) #-,c
            categories=categories.reshape(-1)
            loss=criterion(output,categories)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum_train+=loss.item()
            _,index=output.max(dim=1)

            all_pred.extend(index.cpu().numpy())
            all_categories.extend(categories.cpu().numpy())
            print(f'in step {ind/len(train_loader)}')
        all_pred = np.array(all_pred)
        all_categories = np.array(all_categories)
        accurecy_train = np.mean(all_pred==all_categories) *100
        loss_avg_train = loss_sum_train/len(train_loader)
        f1Score_train =  f1_score(all_categories,all_pred,average='weighted')

        # set pred_need to false to not return labels ,pred
        accurecy_val,loss_avg_val,f1Score_val = evaluate(model,criterion,val_loader,device,False,n_classes)
        
        scheduler.step(loss_avg_val) # step based on avg loss in valdiation data
        logs.append([epoch+1,accurecy_train,loss_avg_train,f1Score_train,accurecy_val,loss_avg_val,f1Score_val])
        if loss_avg_val < best_loss:
            update_checkpint(epoch+1)
            best_loss = loss_avg_val
            print(f"New Best Model found at epoch {epoch+1}")

        print(f"epoch {epoch+1}/{n_epoch}")
        print("Train Resault")
        print(f'accurecy ->{accurecy_train}')
        print(f'loss_avg ->{loss_avg_train}')
        print(f'f1-score ->{f1Score_train}')
        print('==========================================')
        print("Validation Resault")
        print(f'accurecy ->{accurecy_val}')
        print(f'loss_avg ->{loss_avg_val}')
        print(f'f1-score ->{f1Score_val}\n')

    print(f"Loading the best model from epoch {checkpoint['epoch']}")
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict']) 
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    df=pd.DataFrame(logs,columns=['epoch','accurecy_train','loss_avg_train','f1Score_train','accurecy_val','loss_avg_val','f1Score_val'])
    df.to_csv(f'logs/{baseline}_progress.csv',index=False,float_format='%.2f')    

    torch.save(checkpoint,f'checkpoints/{baseline}_best_model_checkpoint.pth')


with open('config/b3_player_classifier.yaml','r') as file:
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
model=B3_Player_Classifier(num_player_actions)
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
train('b3_player_classifier',model,criterion,optimizer,scheduler,train_loader,val_loader,n_epoch,device,num_player_actions)



# Test
print(f"\n--- Test Results ---")
# set pred_need = true to get labels,pred
accurecy_test,loss_avg_test,f1Score_test,all_labels,all_pred = evaluate(model,criterion,test_loader,device,True,num_player_actions)
print('==========================================')
print(f'accurecy ->{accurecy_test}')
print(f'loss_avg ->{loss_avg_test}')
print(f'f1-score ->{f1Score_test}\n')
        
# final_report = Final_Report('b3_player_classifier',all_labels,all_pred)
# print("Create Report in 'outputs/b3_player_classifier'")
# final_report.creat_report()
# print("Create confusion_matrix in 'outputs/b3_player_classifier'")
# final_report.create_confusion_matrix()