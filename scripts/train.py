import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from scripts.eval import evaluate
import logging
import torch.nn as nn

logger=logging.getLogger(__name__)

def train(model,criterion,optimizer,scheduler,train_loader,val_loader,n_epoch,device,checkpoint_path,ind_step,early_stop,n_frozen_layers=0):
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
        for i in range(n_frozen_layers):
           target_model.backbone[i].eval() 
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

            if ind%ind_step==0:
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
        
        if no_update>early_stop:
            logger.warning(f"Early stopping triggered at epoch {epoch+1}\n")
            break
    logger.info(f"Best Model found at epoch {choosen_epoch}\n")
    
