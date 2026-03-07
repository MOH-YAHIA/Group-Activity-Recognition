import numpy as np
import torch 
from sklearn.metrics import f1_score

def evaluate(model,criterion,loader,device,pred_need):
    '''
    pred_need (bool): return labels and pred
    '''
    all_pred=[]
    all_labels=[]
    loss_sum=0
    model.eval()
    with torch.no_grad():
        for whole_frames,cropped_frames,labels in loader:
            whole_frames,cropped_frames,labels=whole_frames.to(device),cropped_frames.to(device),labels.to(device)
            output=model(whole_frames,cropped_frames)
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
