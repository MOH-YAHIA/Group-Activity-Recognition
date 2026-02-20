import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import logging
from utils.logger import setup_logger
from scripts.train import train
from scripts.eval import evaluate
from scripts.final_report import Final_Report
from utils.person_level_dataset import VolleyballPersonDataset
from models.b6 import B6
from models.b3_player_classifier import B3_Player_Classifier

os.makedirs('logs',exist_ok=True)
log_path='logs/b6_progress.log'
setup_logger(log_path)
logger = logging.getLogger(__name__)

with open('config/b6.yaml','r') as file:
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
train_dataset=VolleyballPersonDataset(videos_root,annot_root,train_ids,one_frame=False,player_label=False,train=True)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory)

val_dataset=VolleyballPersonDataset(videos_root,annot_root,val_ids,one_frame=False,player_label=False,train=False)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

test_dataset=VolleyballPersonDataset(videos_root,annot_root,test_ids,one_frame=False,player_label=False,train=False)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone=B3_Player_Classifier(num_player_actions)
backbone.load_state_dict(torch.load('/kaggle/input/datasets/myahiia/b3-player-classifier-dataset/Group-Activity-Recognition/checkpoints/b3_player_classifier_best_model_checkpoint.pth',map_location=device,weights_only=True)['model_state_dict'])
model=B6(backbone,num_group_actions)
model=model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=conf_dict['scheduler']['mode'], factor=conf_dict['scheduler']['factor'], patience=conf_dict['scheduler']['patience'])

# Train
os.makedirs('checkpoints',exist_ok=True)
checkpoint_path='checkpoints/b6_best_model_checkpoint.pth'
train(model,criterion,optimizer,scheduler,train_loader,val_loader,n_epoch,device,checkpoint_path,50,3,8)


# Test
logger.info(f"Test")
# set pred_need = true to get labels,pred
accurecy_test,loss_avg_test,f1Score_test,all_labels,all_pred = evaluate(model,criterion,test_loader,device,True)
logger.info(f'Loss : {loss_avg_test:.4f}')
logger.info(f'ACC  : {accurecy_test:.2f} %')
logger.info(f'F1   : {f1Score_test:.2f} %\n')

output_path='outputs/B6'      
os.makedirs(output_path,exist_ok=True)
final_report = Final_Report(output_path,all_labels,all_pred,for_group=True)
logger.info(f"Create Report in {output_path}")
final_report.creat_report()
logger.info(f"Create Confusion Matrix in {output_path}")
final_report.create_confusion_matrix()
        
