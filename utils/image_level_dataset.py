import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import random
from utils.base_dataset import BaseDataset

class VolleyballImageDataset(BaseDataset):
    def __init__(self,videos_root,annot_root,allowed_ids,one_frame,train):
        super().__init__(videos_root,annot_root,allowed_ids,one_frame,train)

        # we will use resnet50 witch trained on data with these statistics 
        # crop to 224*224, normalize
        # JPEG files are not all created equal. depending on how they were saved, .jpg can be: RGB, grayscale
        # By calling .convert('RGB'), we enforce a shape invariant. it ensures that every single tensor that enters your model has exactly 3 channels
        
        # Training: Standardization + Augmentation
        self.train_transform_image = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)), # Use RandomCrop for Train to add variety
            # Color Jitter: Invariant to gym lighting
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            # Gaussian Blur: Invariant to fast camera motion
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Validation: Standardization ONLY
        self.val_transform_image = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)), # Use CenterCrop for Val to be consistent
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_whole_frame(self,frame,seed):
            random.seed(seed)
            torch.manual_seed(seed)
            if self.train:
                frame=self.train_transform_image(frame)
            else:
                frame=self.val_transform_image(frame)  

            return frame #3*224*224
    

    # we work in image level, for each clip we need 9 frames
    def __getitem__(self, index):
        item=self.samples[index]
        whole_frames=[]
        seed = random.randint(0,2**16)

        # make sure that we process the frames in order
        sorted_frames_id=sorted(item['frame_boxes_dct'].keys(),key=lambda x : int(x))
        if self.one_frame:
            frame_path = os.path.join(self.videos_root,item['video_id'],item['clip_id'],f'{str(sorted_frames_id[4])}.jpg')
            frame=Image.open(frame_path).convert('RGB')

            whole_frames.append(self.get_whole_frame(frame,seed))
        else:
            for frame_id in sorted_frames_id:
                frame_path = os.path.join(self.videos_root,item['video_id'],item['clip_id'],f'{str(frame_id)}.jpg')
                frame=Image.open(frame_path).convert('RGB')

                whole_frames.append(self.get_whole_frame(frame,seed))
        whole_frames=torch.stack(whole_frames,dim=0) #list of tensors -> tensor with dim 9*3*224*224
        label=torch.tensor(self.team_action_dct[item['category']]) #1

        return whole_frames,label
    
 