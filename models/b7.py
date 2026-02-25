import torch,torchvision
import torch.nn as nn
import torchvision.models as models

class B7(nn.Module):
    def __init__(self,backbone,n_group_actions):
        super(B7,self).__init__()
        #B3 player
        self.backbone=backbone.backbone
        for child in list(self.backbone.children())[:7]:
            for param in child.parameters():
                param.requires_grad = False
        self.lstm1=backbone.lstm
       
        self.lstm2=nn.LSTM(input_size=512+512,hidden_size=512,num_layers=1,batch_first=True)
        self.classifier=nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,n_group_actions)
        )

    def forward(self,X):
        # B,9,12,C,W,H
        B,F,P,C,W,H = X.shape
        X=X.view(B*F*P,C,W,H)
        X=self.backbone(X) #B*F*P,2048,1,1
        X=X.view(B,F,P,2048)
        X=X.permute(0,2,1,3).contiguous() #B,P,F,2048
        
        out,(h,c)=self.lstm1(X.view(B*P,F,2048)) #B*P,F,512
        out=out.view(B,P,F,512)
        
        out_t1,_=out[:,:6,:,:].max(dim=1) #B,F,512
        out_t2,_=out[:,6:,:,:].max(dim=1) #B,F,512

        out=torch.concat((out_t1,out_t2),dim=2) #B,F,512+512

        out,(h,c)=self.lstm2(out) #B,F,512
        out=out[:,-1,:] #B,512

        out=self.classifier(out)
        return out #B,8



        
