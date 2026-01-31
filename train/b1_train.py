import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.image_level_dataset import VolleyballImageDataset
from models.b1 import B1
from scripts.train import train
from scripts.eval import evaluate
# Dataset path
annot_root=r"D:\track\Deep learning\cskill\slides\05 Volleyball Project\sample data\volleyball_tracking_annotation"
videos_root=r"D:\track\Deep learning\cskill\slides\05 Volleyball Project\videos_g10"

# Split
train_ids = ['1','3','6','7','10','13','15','16','18','22','23','31','32','36','38','39','40','41','42','48','50','52','53','54']
val_ids = ['0','2','8','12','17','19','24','26','27','28','30','33','46','49','51']
test_ids = ['4','5','9','11','14','20','21','25','29','34','35','37','43','44','45','47']

# DataLoaders
train_dataset=VolleyballImageDataset(videos_root,annot_root,train_ids)
train_loader=DataLoader(train_dataset,batch_size=8,shuffle=True)

val_dataset=VolleyballImageDataset(videos_root,annot_root,val_ids)
val_loader=DataLoader(val_dataset,batch_size=8,shuffle=False)

test_dataset=VolleyballImageDataset(videos_root,annot_root,test_ids)
test_loader=DataLoader(test_dataset,batch_size=8,shuffle=False)

# Hyperparameters
n_epoch = 5
lr = 1e-4
batch_size = 8

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=B1(num_group_actions=8)
model=model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

#  Kaggle use 2 GPU   
if torch.cuda.device_count() > 1:
    print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
    # This is the "Magic" line for T4 x2
    model = nn.DataParallel(model)


# Train
train(model,criterion,optimizer,scheduler,train_loader,val_loader,n_epoch,device)



# Test
print(f"\n--- Test Results ---")
accurecy_test,loss_avg_test,f1Score_test = evaluate(model,criterion,test_loader,device)
print('==========================================')
print(f'accurecy ->{accurecy_test}')
print(f'loss_avg ->{loss_avg_test}')
print(f'f1-score ->{f1Score_test}\n')
        