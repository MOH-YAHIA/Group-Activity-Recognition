import torch,torchvision
import torch.nn as nn
import torchvision.models as models

class B7(nn.Module):
    def __init__(self,backbone,n_group_actions):
        super(B7,self).__init__()
        #B3 player
        self.backbone=backbone.backbone
        for child in list(self.backbone.children())[:8]:
            for param in child.parameters():
                param.requires_grad = False
        self.lstm1=backbone.lstm
        #describe each player with resnet vis + temporal describe 
        self.lstm2=nn.LSTM(input_size=2048+512,hidden_size=512,num_layers=1,batch_first=True)
        self.classifier=nn.Sequential(
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,n_group_actions)
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
        # static vis for each person from resnet + temporal 
        # X(B,P,F,2048) out(B,P,F,512) 
        combined=torch.cat((X,out),dim=-1) #B,P,F,2048+512
        combined,_=combined.max(dim=1) #B,F,2048+512

        combined,(h,c)=self.lstm2(combined) #B,F,512
        combined=combined[:,-1,:] #B,512

        combined=self.classifier(combined)
        return combined #B,8



        
