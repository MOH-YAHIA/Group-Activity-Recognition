import torch,torchvision
import torch.nn as nn
import torchvision.models as models

class B5_Group_Classifier_Temporal(nn.Module):
    def __init__(self,backbone,n_group_actions):
        super(B5_Group_Classifier_Temporal,self).__init__()
        self.backbone=backbone.backbone
        self.lstm=backbone.lstm
        for child in list(self.backbone.children())[:8]:
            for param in child.parameters():
                param.requires_grad = False
        #for param in self.lstm.parameters():
            #param.requires_grad = False

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
        #with torch.no_grad():
        X=self.backbone(X) #B*F*P,2048,1,1
        X=X.view(B,F,P,2048)
        X=X.permute(0,2,1,3).contiguous() #B,P,F,2048
        X=X.view(B*P,F,2048)
        out,(h,c)=self.lstm(X) #B*P,F,512
        out=out[:,-1,:].reshape(B,P,512) 
        out,_=out.max(dim=1) #B,512
        out=self.classifier(out)
        return out #B,8



        
