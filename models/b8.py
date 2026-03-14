import torch
import torch.nn as nn

class B8(nn.Module):
    def __init__(self,backbone_image,backbone_player,n_group_actions):
        super(B8,self).__init__()
        
        self.backbone_image=nn.Sequential(*list(backbone_image.model.children())[:-1])
        self.backbone_player=backbone_player.backbone
        self.lstm1=backbone_player.lstm

        for param in self.backbone_image.parameters():
            param.requires_grad = False
        for param in self.backbone_player.parameters():
            param.requires_grad = False
        for param in self.lstm1.parameters():
            param.requires_grad = False
        
        # Projection to reduce 2048 -> 512
        self.vis_proj_image = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.vis_proj_player = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.lstm2=nn.LSTM(input_size=2560,hidden_size=512,num_layers=1,batch_first=True)
        self.classifier=nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,n_group_actions)
        )

    def forward(self,X_image,X_player):
        # B,9,C,W,H    # B,9,12,C,W,H 
        B,F,P,C,W,H = X_player.shape
        X_image=X_image.view(B*F,C,W,H)
        X_player=X_player.view(B*F*P,C,W,H)

        with torch.no_grad():
            X_image=self.backbone_image(X_image) #B*F,2048,1,1

            X_player=self.backbone_player(X_player) #B*F*P,2048,1,1
            X_player=X_player.view(B,F,P,2048)
            X_player=X_player.permute(0,2,1,3).contiguous() #B,P,F,2048
            
            out_temp_player,(h,c)=self.lstm1(X_player.view(B*P,F,2048)) #B*P,F,512
            out_temp_player=out_temp_player.view(B,P,F,512)
        
        # static vis for each frame from resnet B1
        out_vis_image=self.vis_proj_image(X_image.view(-1,2048)).view(B,F,512) #B,F,512
        # static vis for each person from resnet B3 
        out_vis_player=self.vis_proj_player(X_player.view(-1,2048)).view(B,P,F,512) #B,P,F,512

        # out_vis_player(B,P,F,512) out_temp_player(B,P,F,512) 
        combined_player=torch.concat((out_vis_player,out_temp_player),dim=-1) #B,P,F,512+512

        out_t1,_=combined_player[:,:6,:,:].max(dim=1) #B,F,1024
        out_t2,_=combined_player[:,6:,:,:].max(dim=1) #B,F,1024


        out=torch.concat((out_t1,out_t2),dim=2) #B,F,2048
        #description for players + whole frame
        out=torch.concat((out,out_vis_image),dim=2) #B,F,2560

        out,(h,c)=self.lstm2(out) #B,F,512
        out=out[:,-1,:] #B,512

        out=self.classifier(out)
        return out #B,8
