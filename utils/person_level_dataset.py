import torch
import os
from PIL import Image
import torchvision.transforms as transforms

from utils.base_dataset import BaseDataset

class VolleyballPersonDataset(BaseDataset):
    def __init__(self,videos_root,annot_root,allowed_ids,one_frame=False):
        super().__init__(videos_root,annot_root,allowed_ids,one_frame)

        self.preprocess = transforms.Compose([
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

        def crop_fun(video_id,clip_id,frame_id,frame_boxes):
            img_path = os.path.join(self.videos_root,video_id,clip_id,f'{str(frame_id)}.jpg')
            img = Image.open(img_path).convert('RGB')
            cropped_boxes=[] 
            players_category=[]
            # for each player 
            for box in frame_boxes:
                x1, y1, x2, y2 = box.box
                crop=img.crop((x1, y1, x2, y2))
                crop=self.preprocess(crop) #3*224*224
                cropped_boxes.append(crop)
                players_category.append(self.player_action_dct[box.category])
            
            # not every player appears in the clip 
            # after that , each frame can contain less than 12 player 
            # fix this by adding zeroes players
            # with null label
            while len(cropped_boxes)<12:
                cropped_boxes.append(torch.zeros_like(cropped_boxes[0]))
                players_category.append(-1)

            #cropped_boxes 12*3*224*224
            #players_category 12
            frames.append(torch.stack(cropped_boxes)) 
            categories.append(torch.tensor(players_category))
        
        if self.one_frame:
            crop_fun(item['video_id'],item['clip_id'],list(item['frame_boxes_dct'].keys())[4],list(item['frame_boxes_dct'].values())[4])
        else:
            for frame_id,frame_boxes in item['frame_boxes_dct'].items():
                crop_fun(item['video_id'],item['clip_id'],frame_id,frame_boxes)

        frames = torch.stack(frames) # 9*12*3*224*224
        categories = torch.stack(categories) # 9*12
        label = torch.tensor(self.team_action_dct[item['category']]) 

        return frames, categories, label


