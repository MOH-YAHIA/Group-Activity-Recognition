import numpy as np
from sklearn.metrics import f1_score
from scripts.eval import evaluate

def train(model,criterion,optimizer,scheduler,train_loader,val_loader,n_epoch,device):    
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

        
        accurecy_val,loss_avg_val,f1Score_val = evaluate(model,criterion,val_loader,device)
        
        scheduler.step(loss_avg_val)
        
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
        


