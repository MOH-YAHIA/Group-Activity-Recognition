import torch
import os
from PIL import Image
import torchvision.transforms as transforms

from utils.base_dataset import BaseDataset

class VolleyballImageDataset(BaseDataset):
    def __init__(self,videos_root,annot_root,allowed_ids,one_frame,train):
        super().__init__(videos_root,annot_root,allowed_ids,one_frame)

        self.train=train
        # we will use resnet50 witch trained on data with these statistics 
        # crop to 224*224, normalize

        # Training: Standardization + Augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)), # Use RandomCrop for Train to add variety
            # Color Jitter: Invariant to gym lighting
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            # Gaussian Blur: Invariant to fast camera motion
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Validation: Standardization ONLY
        self.val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)), # Use CenterCrop for Val to be consistent
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
            if self.train:
                frame=self.train_transform(frame)
            else:
                frame=self.val_transform(frame)  

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
    
 