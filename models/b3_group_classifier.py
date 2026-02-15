import torch
import torch.nn as nn

class B3_Group_Classifier(nn.Module):
    def __init__(self,backbone,num_group_actions):
        super(B3_Group_Classifier,self).__init__()
        # remove .fc 
        self.backbone=nn.Sequential(*list(backbone.resnet.children())[:-1])
        for child in list(self.backbone.children())[:7]:
            for param in child.parameters():
                param.requires_grad = False
        self.classifier=nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_group_actions)
        )

    def forward(self,X):
        # X -> B,1,12,3,224,224
        B,F,P,C,W,H=X.shape
        X=X.view(B*F*P,C,W,H)
        X=self.backbone(X) #B*F*P,2048,1,1
        X=torch.flatten(X,1) #B*F*P,2048
        X=X.view(B*F,P,-1)
        X,_=X.max(dim=1) #B*F,2048

        out=self.classifier(X) #B*F,8  F=1
        return out
