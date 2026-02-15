import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import random
from utils.base_dataset import BaseDataset

class VolleyballPersonDataset(BaseDataset):
    def __init__(self,videos_root,annot_root,allowed_ids,one_frame,player_label,train):
        super().__init__(videos_root,annot_root,allowed_ids,one_frame)
        self.player_label=player_label
        self.train=train
        self.train_transform = transforms.Compose([
            # 1. Flip: A spike is a spike regardless of direction
            transforms.RandomHorizontalFlip(p=0.5),
            # 2. Color: Handle different jersey colors and gym lighting
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            # 3. Standardization (Invariant)
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # 4. Occlusion: Handle players blocking each other
            # Note: RandomErasing must come AFTER ToTensor
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            # 5. Normalization (Invariant)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.val_transform = transforms.Compose([
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
   
        
    
    # in each frame we need tensor that contains tesnsor for each player crop 
    # 9(frames)*12(player)*3(channel)*224(W)*224(H)
    # category for each palyer in each frame 9*12
    # category for group in the whole clip  1
    def __getitem__(self, index):
        item=self.samples[index]
        frames=[]
        categories=[]
        seed = random.randint(0,2**16)

        def crop_fun(video_id,clip_id,frame_id,frame_boxes):
            img_path = os.path.join(self.videos_root,video_id,clip_id,f'{str(frame_id)}.jpg')
            img = Image.open(img_path).convert('RGB')
            cropped_boxes=[] 
            players_category=[]
            # for each player 
            for ind,box in enumerate(frame_boxes):
                x1, y1, x2, y2 = box.box
                crop=img.crop((x1, y1, x2, y2))
                # for each player we need the same transormation along the 9 frames
                # so we have unique seed for each player
                random.seed(seed+ind)
                torch.manual_seed(seed+ind)
                if self.train:
                    crop=self.train_transform(crop)
                else:
                    crop=self.val_transform(crop) 
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

            #cropped_boxes 12*3*224*224
            #players_category 12
            frames.append(torch.stack(cropped_boxes)) 
            categories.append(torch.tensor(players_category))
        
        # make sure that we process the frames in order
        sorted_frames_id=sorted(item['frame_boxes_dct'].keys(),key=lambda x : int(x))
        if self.one_frame:
            crop_fun(item['video_id'],item['clip_id'],sorted_frames_id[4],item['frame_boxes_dct'][sorted_frames_id[4]])
        else:
            for frame_id in sorted_frames_id:
                crop_fun(item['video_id'],item['clip_id'],frame_id,item['frame_boxes_dct'][frame_id])

        frames = torch.stack(frames) # 9*12*3*224*224
        categories = torch.stack(categories) # 9*12
        label = torch.tensor(self.team_action_dct[item['category']]) 

        if not self.player_label:
            return frames,label
        return frames, categories, label


