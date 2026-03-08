import numpy as np
import torch
from sklearn.metrics import f1_score
from scripts.eval_b7_8 import evaluate
import logging
import os 

logger=logging.getLogger(__name__)

def train(model,criterion,optimizer,scheduler,train_loader,val_loader,n_epoch,device,checkpoint_path,ind_step,early_stop,trained_model=None):
    best_loss=float('inf') 
    no_update=0
    choosen_epoch=0
    target_model=model.module if isinstance(model, torch.nn.DataParallel) else model

    def update_checkpint(epoch,checkpoint_path):
        checkpoint = {
        'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss
        }
        torch.save(checkpoint,checkpoint_path)

    if trained_model is not None:
        loaded_checkpoint=torch.load(trained_model,map_location=device,weights_only=True)
        target_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(loaded_checkpoint['scheduler_state_dict'])
        choosen_epoch=loaded_checkpoint['epoch']
        best_loss=loaded_checkpoint['best_loss']
        optimizer.param_groups[0]['lr']/=10
        logger.info(f'Continue learning from epoch {choosen_epoch+1}')

    for epoch in range(choosen_epoch,n_epoch):
        logger.info(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")
        loss_sum_train=0
        all_pred=[]
        all_labels=[]
        model.train()
        target_model.backbone_image.eval()
        target_model.backbone_player.eval()
        target_model.lstm1.eval()

        
        for ind,(whole_frames, cropped_frames, labels) in enumerate(train_loader):
            whole_frames,cropped_frames,labels=whole_frames.to(device),cropped_frames.to(device),labels.to(device)
            output=model(whole_frames,cropped_frames)
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

        update_checkpint(epoch+1,os.path.join(checkpoint_path,'latest_model_checkpoint.pth'))
        if loss_avg_val < best_loss:
            update_checkpint(epoch+1,os.path.join(checkpoint_path,'best_model_checkpoint.pth'))
            best_loss = loss_avg_val
            no_update = 0
            choosen_epoch = epoch+1
            logger.info(f"New Best Model found at epoch {epoch+1}\n")
        else:
            no_update+=1
        
        if no_update>=early_stop:
            logger.warning(f"Early stopping triggered at epoch {epoch+1}\n")
            break
    logger.info(f"Best Model found at epoch {choosen_epoch}\n")
    
