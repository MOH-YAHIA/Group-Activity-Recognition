import numpy as np
import torch
import torch.nn as nn
from models.b1 import B1
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from utils.image_level_dataset import VolleyballImageDataset

# Dataset path
annot_root=r"D:\track\Deep learning\cskill\slides\05 Volleyball Project\sample data\volleyball_tracking_annotation"
videos_root=r"D:\track\Deep learning\cskill\slides\05 Volleyball Project\videos_g10"


train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                 "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]

# DataLoaders
train_dataset=VolleyballImageDataset(videos_root,annot_root,train_ids)
train_loader=DataLoader(train_dataset,batch_size=8,shuffle=True)

val_dataset=VolleyballImageDataset(videos_root,annot_root,val_ids)
val_loader=DataLoader(val_dataset,batch_size=8,shuffle=True)

# Hyperparameters
n_epoch = 5
lr = 1e-5
batch_size = 8

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=B1(num_group_actions=8)
model=model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Train
for epoch in range(n_epoch):
    loss_sum=0
    for ind,(imgs,labels) in enumerate(train_loader):
        imgs,labels=imgs.to(device),labels.to(device)
        output=model(imgs)
        loss=criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum+=loss.item()
    print(f'epoch:{epoch}, loss:{loss_sum/len(train_loader)}')



# Validation
print(f"\n--- Validation Results ---")

all_pred=[]
all_labels=[]
loss_sum=0
model.eval()
with torch.no_grad():
    for imgs,labels in val_loader:
        imgs,labels=imgs.to(device),labels.to(device)
        output=model(imgs)
        loss=criterion(output,labels)
        loss_sum+=loss.item()
        value,index=output.max(dim=1)

        all_pred.extend(index.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_pred = np.array(all_pred)
all_labels = np.array(all_labels)

accurecy = np.mean(all_pred==all_labels) *100
loss = loss_sum / len(val_loader)
f1Score =  f1_score(all_labels,all_pred,average='weighted')


print(f'accurecy -> {accurecy}')
print(f'loss -> {loss}')
print(f'f1_score -> {f1Score}')

