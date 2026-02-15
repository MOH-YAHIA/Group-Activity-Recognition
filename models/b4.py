import torch,torchvision
import torch.nn as nn
import torchvision.models as models

class B4(nn.Module):
    def __init__(self,backbone,num_group_actions):
        super(B4,self).__init__()
        self.backbone=nn.Sequential(*list(backbone.model.children())[:-1])
        for parm in self.backbone.parameters():
            parm.requires_grad = False
        self.lstm=nn.LSTM(input_size=2048,hidden_size=512,num_layers=1,batch_first=True)
        self.classifier=nn.Linear(512,num_group_actions)

    def forward(self,X):
        # X -> B,9,3,224,224
        B,F,C,W,H=X.shape
        X=X.view(B*F,C,W,H)
        with torch.no_grad():
            X=self.backbone(X) #B*F,2048
        X=X.view(B,F,2048) #batch,seq_len,input_size
        out,(h,c)=self.lstm(X) #out -> B,F,512
        out=out[:,-1,:] #B,512
        out=self.classifier(out) #B,8
        return out
        


        