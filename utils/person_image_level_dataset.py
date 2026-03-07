import torch
import random
import os
from PIL import Image
from utils.person_level_dataset import VolleyballPersonDataset
from utils.image_level_dataset import VolleyballImageDataset

class VolleyballPersonImageDataset(VolleyballImageDataset,VolleyballPersonDataset):
    def __init__(self,videos_root,annot_root,allowed_ids,one_frame,train):
        super().__init__(videos_root,annot_root,allowed_ids,one_frame,train)


    def __getitem__(self, index):
        item=self.samples[index]
        cropped_frames=[]
        whole_frames=[]
        seed = random.randint(0,2**16)

        # make sure that we process the frames in order
        sorted_frames_id=sorted(item['frame_boxes_dct'].keys(),key=lambda x : int(x))

        if self.one_frame:
            frame_path = os.path.join(self.videos_root,item['video_id'],item['clip_id'],f'{str(sorted_frames_id[4])}.jpg')
            frame=Image.open(frame_path).convert('RGB')

            whole_frames.append(self.get_whole_frame(frame,seed))
            cropped_boxes,_=self.get_cropped_frame(frame,item['frame_boxes_dct'][sorted_frames_id[4]],seed)
            cropped_frames.append(cropped_boxes)
        else:
            for frame_id in sorted_frames_id:
                frame_path = os.path.join(self.videos_root,item['video_id'],item['clip_id'],f'{str(frame_id)}.jpg')
                frame = Image.open(frame_path).convert('RGB')
            
                whole_frames.append(self.get_whole_frame(frame,seed))
                cropped_boxes,_=self.get_cropped_frame(frame,item['frame_boxes_dct'][frame_id],seed)
                cropped_frames.append(cropped_boxes)


        whole_frames = torch.stack(whole_frames,dim=0) # 9*3*224*224
        cropped_frames = torch.stack(cropped_frames,dim=0) # 9*12*3*224*224
        label = torch.tensor(self.team_action_dct[item['category']]) 
        
        return whole_frames, cropped_frames, label
