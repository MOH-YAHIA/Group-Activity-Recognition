import numpy as np
import torch 
from sklearn.metrics import f1_score

def evaluate(model,criterion,loader,device,pred_need):
    all_pred=[]
    all_labels=[]
    loss_sum=0
    model.eval()
    with torch.no_grad():
        for imgs,labels in loader:
            imgs,labels=imgs.to(device),labels.to(device)
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