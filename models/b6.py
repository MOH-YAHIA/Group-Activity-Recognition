import torch,torchvision
import torch.nn as nn
import torchvision.models as models

class B6(nn.Module):
    def __init__(self,backbone,n_group_actions):
        super(B6,self).__init__()
        self.backbone=nn.Sequential(*list(backbone.resnet.children())[:-1])
        # freez all layers in backbone
        for child in list(self.backbone.children())[:7]:
            for param in child.parameters():
                param.requires_grad = False
        self.lstm=nn.LSTM(input_size=2048,hidden_size=512,num_layers=1,batch_first=True)
        self.classifier=nn.Sequential(
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,n_group_actions)
        )

    def forward(self,X):
        # X -> B,9,12,3,224,224
        B,F,P,C,W,H=X.shape
        X=X.view(B*F*P,C,W,H)
        X=self.backbone(X) #B*F*P,2048,1,1
        X=X.view(B,F,P,2048) 
        X,_=X.max(dim=2) #B,F,2048
        #batch,seq_len,input_size
        out,(h,c)=self.lstm(X) #out -> B,F,512
        out=out[:,-1,:] #B,512
        out=self.classifier(out) #B,8
        return out
        


        