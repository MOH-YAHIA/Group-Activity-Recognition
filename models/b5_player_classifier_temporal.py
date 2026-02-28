import torch,torchvision
import torch.nn as nn
import torchvision.models as models

class B5_Player_Classifier_Temporal(nn.Module):
    def __init__(self,n_player_actions):
        super(B5_Player_Classifier_Temporal,self).__init__()
        self.backbone=nn.Sequential(*list(models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT).children())[:-1])

        self.lstm=nn.LSTM(input_size=2048,hidden_size=512,num_layers=1,batch_first=True)
        self.classifier=nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,n_player_actions)
        )

    def forward(self,X):
        # B,9,12,C,W,H
        B,F,P,C,W,H = X.shape
        X=X.view(B*F*P,C,W,H)
        X=self.backbone(X) #B*F*P,2048,1,1
        X=X.view(B,F,P,2048)
        X=X.permute(0,2,1,3).contiguous() #B,P,F,2048
        X=X.view(B*P,F,2048)
        out,(h,c)=self.lstm(X) #B*P,F,512
        out=out[:,-1,:] #B*P,512
        out=self.classifier(out) #B*P,9
        out=out.view(B,P,9)
        return out #B,P,9



        
