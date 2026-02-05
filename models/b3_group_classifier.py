import torch,torchvision
import torch.nn as nn
import torchvision.models as models

class B3_Group_Classifier(nn.Module):
    def __init__(self,backbone,num_player_actions,num_group_actions):
        super(B3_Group_Classifier,self).__init__()
        self.backbone=backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier=nn.Linear(num_player_actions,num_group_actions)
    def forward(self,X):
        # X -> B,1,12,3,224,224
        B,F,P,C,W,H=X.shape
        X=self.backbone(X) #B,F,P,9
        X=X.view(B*F,P,9) 
        X,_=X.max(dim=1) #B,9

        return self.classifier(X) #B,8