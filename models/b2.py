import torch,torchvision
import torch.nn as nn
import torchvision.models as models

class B2(nn.Module):
    def __init__(self,num_group_actions):
        super(B2,self).__init__()
        self.resnet=models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        in_fc=self.resnet.fc.in_features
        self.backbone=nn.Sequential(*list(self.resnet.children())[:-1])
        self.classifier=nn.Linear(in_fc,num_group_actions)

    def forward(self,X):
        # X -> B,12,3,224,224
        B,P,C,W,H=X.shape
        # resnet take tensor of 4 -> treate the 12 players as batches 
        # so for B=2 the batch_size would be 24
        X=X.view(B*P,C,W,H)
        out=self.backbone(X) #B*P,2048,1,1
        out=out.view(B,P,-1) #B,P,2048
        out,_=out.max(dim=1) #B,2048
        return self.classifier(out) 