import torch,torchvision
import torch.nn as nn
import torchvision.models as models

class B3_Player_Classifier(nn.Module):
    def __init__(self,num_player_actions):
        super(B3_Player_Classifier,self).__init__()
        self.resnet=models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        in_fc=self.resnet.fc.in_features
        self.resnet.fc=nn.Linear(in_fc,num_player_actions)

    def forward(self,X):
        # X -> B,9,12,3,224,224
        B,F,P,C,W,H=X.shape
        # resnet take tensor of 4 -> treate the frames and the 12 players as batches 
        # so for B=2 the batch_size would be 24
        X=X.view(B*F*P,C,W,H)
        out=self.resnet(X) #B*F*P,9  #9 actions
        out=out.view(B,F,P,-1)
        return out