import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import random
from utils.base_dataset import BaseDataset

class VolleyballPersonDataset(BaseDataset):
    def __init__(self,videos_root,annot_root,allowed_ids,one_frame,train,player_label=False):
        super().__init__(videos_root,annot_root,allowed_ids,one_frame,train)
        self.player_label=player_label
        self.train_transform_person = transforms.Compose([
            # 1. Flip(player): A spike is a spike regardless of direction
            # comment flip in group classifier
            # transforms.RandomHorizontalFlip(p=0.5),

            # 2. Color: Handle different jersey colors and gym lighting
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            
            # 3. Blur: Handle fast motion tracking of the ball/play
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      
        ])
        self.val_transform_person = transforms.Compose([
            #we will clip each frame, no need for center clip
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
       
        self.player_action_dct = {
            'waiting': 0, 'setting': 1, 'digging': 2, 
            'falling': 3,'spiking': 4, 'blocking': 5,
            'jumping': 6, 'moving': 7, 'standing': 8
        }
   
        
    def get_cropped_frame(self,frame,frame_boxes,seed):
            cropped_boxes=[] 
            players_category=[]
            # for each player 
            for ind,box in enumerate(frame_boxes):
                x1, y1, x2, y2 = box.box
                crop=frame.crop((x1, y1, x2, y2))
                random.seed(seed)
                torch.manual_seed(seed)
                if self.train:
                    crop=self.train_transform_person(crop)
                else:
                    crop=self.val_transform_person(crop) 
                #crop 3*224*224
                cropped_boxes.append(crop)
                players_category.append(self.player_action_dct[box.category])
            
            # not every player appears in the clip 
            # after that , each frame can contain less than 12 player 
            # fix this by adding zeroes players
            # with null label
            while len(cropped_boxes)<12:
                cropped_boxes.append(torch.zeros(3,224,224))
                players_category.append(-1)

            return torch.stack(cropped_boxes,dim=0) , torch.tensor(players_category)
            #cropped_boxes 12*3*224*224
            #players_category 12

    # in each frame we need tensor that contains tesnsor for each player crop 
    # 9(frames)*12(player)*3(channel)*224(W)*224(H)
    # category for each palyer in each frame 9*12
    # category for group in the whole clip  1
    def __getitem__(self, index):
        item=self.samples[index]
        cropped_frames=[]
        categories=[]
        seed = random.randint(0,2**16)
   
        # make sure that we process the frames in order
        sorted_frames_id=sorted(item['frame_boxes_dct'].keys(),key=lambda x : int(x))

        if self.one_frame:
            frame_path = os.path.join(self.videos_root,item['video_id'],item['clip_id'],f'{str(sorted_frames_id[4])}.jpg')
            frame = Image.open(frame_path).convert('RGB')
         
            cropped_boxes,players_category=self.get_cropped_frame(frame,item['frame_boxes_dct'][sorted_frames_id[4]],seed)
            cropped_frames.append(cropped_boxes)
            categories.append(players_category)
        else:
            for frame_id in sorted_frames_id:
                frame_path = os.path.join(self.videos_root,item['video_id'],item['clip_id'],f'{str(frame_id)}.jpg')
                frame = Image.open(frame_path).convert('RGB')
            
                cropped_boxes,players_category=self.get_cropped_frame(frame,item['frame_boxes_dct'][frame_id],seed)                
                cropped_frames.append(cropped_boxes)
                categories.append(players_category)

        cropped_frames = torch.stack(cropped_frames,dim=0) # 9*12*3*224*224
        categories = torch.stack(categories,dim=0) # 9*12
        label = torch.tensor(self.team_action_dct[item['category']]) 
        if not self.player_label:
            return cropped_frames,label
        return cropped_frames, categories, label
