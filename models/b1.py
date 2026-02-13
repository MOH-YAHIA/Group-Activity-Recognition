import torchvision
import torch.nn as nn
import torchvision.models as models


class B1(nn.Module):
    def __init__(self,num_group_actions):
        super(B1,self).__init__()
        self.model=models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        in_fc=self.model.fc.in_features
        self.model.fc=nn.Sequential(
            nn.Linear(in_fc,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_group_actions)
        )

    def forward(self,X):
        # B,3,224,224
        X = X.view(-1,3,224,224)
        out = self.model(X) #B,8
        return out