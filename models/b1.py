import torch,torchvision
import torch.nn as nn
import torchvision.models as models


class B1(nn.Module):
    def __init__(self,num_group_actions):
        super(B1,self).__init__()
        self.model=models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        in_fc=self.model.fc.in_features
        self.model.fc=nn.Linear(in_fc,num_group_actions)

    def forward(self,X):
        return self.model(X)