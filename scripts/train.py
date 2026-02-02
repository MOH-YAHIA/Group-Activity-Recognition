import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from scripts.eval import evaluate

def train(baseline,model,criterion,optimizer,scheduler,train_loader,val_loader,n_epoch,device):    
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
        all_labels=[]
        model.train()
        for ind,(imgs,labels) in enumerate(train_loader):
            imgs,labels=imgs.to(device),labels.to(device)
            output=model(imgs)
            loss=criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum_train+=loss.item()
            _,index=output.max(dim=1)

            all_pred.extend(index.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_pred = np.array(all_pred)
        all_labels = np.array(all_labels)
        accurecy_train = np.mean(all_pred==all_labels) *100
        loss_avg_train = loss_sum_train/len(train_loader)
        f1Score_train =  f1_score(all_labels,all_pred,average='weighted')

        # set pred_need to false to not return labels ,pred
        accurecy_val,loss_avg_val,f1Score_val = evaluate(model,criterion,val_loader,device,False)
        
        scheduler.step(loss_avg_val) # step based on avg loss in valdiation data
        logs.append([epoch+1,accurecy_train,loss_avg_train,f1Score_train,accurecy_val,loss_avg_val,f1Score_val])
        if loss_avg_val < best_loss:
            update_checkpint(epoch)
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

    print(f"Loading the best model from epoch {checkpoint['epoch']+1}")
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict']) 
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    df=pd.DataFrame(logs,columns=['epoch','accurecy_train','loss_avg_train','f1Score_train','accurecy_val','loss_avg_val','f1Score_val'])
    df.to_csv(f'logs/{baseline}_progress.csv',index=False,float_format='%.2f')    

    torch.save(checkpoint,f'outputs/{baseline.upper()}/best_model_checkpoint.pth')

