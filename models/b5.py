import torch,torchvision
import torch.nn as nn
import torchvision.models as models

class B5(nn.Module):
    def __init__(self,backbone,num_group_actions):
        super(B5,self).__init__()
        self.backbone=nn.Sequential(*list(backbone.resnet.children())[:-1])
        for parm in self.backbone.parameters():
            parm.requires_grad = False
        self.lstm=nn.LSTM(input_size=2048,hidden_size=512,num_layers=1,batch_first=True)
        self.classifier=nn.Linear(512,num_group_actions)

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
        


        