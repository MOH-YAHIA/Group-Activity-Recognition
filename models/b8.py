import torch,torchvision
import torch.nn as nn
import torchvision.models as models

class B8(nn.Module):
    def __init__(self,backbone,n_group_actions):
        super(B8,self).__init__()
        self.backbone=backbone.backbone
        self.lstm1=backbone.lstm

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.lstm1.parameters():
            param.requires_grad = False
        
        self.vis_norm=nn.LayerNorm(2048)

        self.lstm2=nn.LSTM(input_size=5120,hidden_size=512,num_layers=1,batch_first=True)
        self.classifier=nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,n_group_actions)
        )

    def forward(self,X):
        # B,9,12,C,W,H
        B,F,P,C,W,H = X.shape
        X=X.view(B*F*P,C,W,H)
        with torch.no_grad():
            X=self.backbone(X) #B*F*P,2048,1,1
            X=X.view(B,F,P,2048)
            X=X.permute(0,2,1,3).contiguous() #B,P,F,2048
            
            out_temp,(h,c)=self.lstm1(X.view(B*P,F,2048)) #B*P,F,512
            out_temp=out_temp.view(B,P,F,512)
            
        # static vis for each person from resnet + temporal 
        out_vis=self.vis_norm(X.view(-1,2048)).view(B,P,F,2048) #B,P,F,2048
        # X(B,P,F,2048) out(B,P,F,512) 
        combined=torch.concat((out_vis,out_temp),dim=-1) #B,P,F,2048+512

        out_t1,_=combined[:,:6,:,:].max(dim=1) #B,F,2560
        out_t2,_=combined[:,6:,:,:].max(dim=1) #B,F,2560

        out=torch.concat((out_t1,out_t2),dim=2) #B,F,2560+2560

        out,(h,c)=self.lstm2(out) #B,F,512
        out=out[:,-1,:] #B,512

        out=self.classifier(out)
        return out #B,8



        
