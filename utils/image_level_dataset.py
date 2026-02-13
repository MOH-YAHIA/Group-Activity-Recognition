import torch
import os
from PIL import Image
import torchvision.transforms as transforms

from utils.base_dataset import BaseDataset

class VolleyballImageDataset(BaseDataset):
    def __init__(self,videos_root,annot_root,allowed_ids,one_frame):
        super().__init__(videos_root,annot_root,allowed_ids,one_frame)

        # we will use resnet50 witch trained on data with these statstics 
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
 
    # we work in image level, for each clip we need middle frame with clip label 
    def __getitem__(self, index):
        item=self.samples[index]
        frames=[]
        def get_frame(video_id,clip_id,frame_id):
            frame_path=os.path.join(self.videos_root,video_id,clip_id,f'{frame_id}.jpg')
            # JPEG files are not all created equal. depending on how they were saved, .jpg can be: RGB, grayscale
            # By calling .convert('RGB'), we enforce a shape invariant. it ensures that every single tensor that enters your model has exactly 3 channels
            frame=Image.open(frame_path).convert('RGB')
            frame=self.preprocess(frame) #return tensor 

            frames.append(frame)

        if self.one_frame:
            middle_frame_id=list(item['frame_boxes_dct'].keys())[4] # 9 frames -> middle = index 4
            get_frame(item['video_id'],item['clip_id'],middle_frame_id)
        else:
            for frame_id,_ in item['frame_boxes_dct'].items():
                get_frame(item['video_id'],item['clip_id'],frame_id)

        frames=torch.stack(frames)        
        label=torch.tensor(self.team_action_dct[item['category']])

        return frames,label
    
 