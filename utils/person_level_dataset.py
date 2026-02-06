import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

from utils.volleyball_annot_loader import load_volleyball_dataset

class VolleyballPersonDataset(Dataset):
    def __init__(self,videos_root,annot_root,allowed_ids,one_frame=False):
        self.videos_root=videos_root
        self.one_frame=one_frame
        self.preprocess = transforms.Compose([
            #we will clip each frame, no need for center clip
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.team_action_dct = {
        'l-pass': 0,     'r-pass': 1,
        'l-spike': 2,    'r_spike': 3,
        'l_set': 4,      'r_set': 5,
        'l_winpoint': 6, 'r_winpoint': 7
        }
        self.player_action_dct = {
            'waiting': 0, 'setting': 1, 'digging': 2, 
            'falling': 3,'spiking': 4, 'blocking': 5,
            'jumping': 6, 'moving': 7, 'standing': 8
        }
        self.annotations_dict=load_volleyball_dataset(videos_root,annot_root)
        self.samples=[]

        for video_id,clips in self.annotations_dict.items():
            if video_id not in allowed_ids:
                continue
            for clip_id,clip in clips.items():
                self.samples.append(
                    {
                        'video_id':video_id,
                        'clip_id':clip_id,
                        'category':clip['category'],
                        'frame_boxes_dct':clip['frame_boxes_dct']
                    }
                )
        
    def __len__(self):
        return len(self.samples)
    
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
            # with standing label
            while len(cropped_boxes)<12:
                cropped_boxes.append(torch.zeros_like(cropped_boxes[0]))
                players_category.append(self.player_action_dct['standing'])

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


    
# annot_root=r"D:\track\Deep learning\cskill\slides\05 Volleyball Project\sample data\volleyball_tracking_annotation"
# videos_root=r"D:\track\Deep learning\cskill\slides\05 Volleyball Project\videos_g10"

# img_da=VolleyballPersonDataset(videos_root,annot_root)

# print(img_da.__len__())
# frames, categories, label=img_da.__getitem__(10)
# print(frames.shape)
# print(categories.shape)
# print(label.shape) 