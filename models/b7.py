import torch,torchvision
import torch.nn as nn
import torchvision.models as models

class B7(nn.Module):
    def __init__(self,backbone,n_group_actions):
        super(B7,self).__init__()

        self.backbone=backbone.backbone
        self.lstm1=backbone.lstm
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.lstm1.parameters():
            param.requires_grad = False

        self.lstm2=nn.LSTM(input_size=512,hidden_size=512,num_layers=1,batch_first=True)
        self.classifier=nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_group_actions)
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
        
        out,_=out.max(dim=1) #B,F,512

        out,(h,c)=self.lstm2(out) #B,F,512
        out=out[:,-1,:] #B,512

        out=self.classifier(out)
        return out #B,8



        
