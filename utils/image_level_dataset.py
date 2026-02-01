import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

from utils.volleyball_annot_loader import load_volleyball_dataset

class VolleyballImageDataset(Dataset):
    def __init__(self,videos_root,annot_root,allowed_ids):
        self.videos_root=videos_root
        # we will use resnet50 witch trained on data with these statstics 
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.team_action_dct = {
        'l-pass': 0,     'r-pass': 1,
        'l-spike': 2,    'r_spike': 3,
        'l_set': 4,      'r_set': 5,
        'l_winpoint': 6, 'r_winpoint': 7
        }
        # get the annotation for the whole dataset in dictionary
        self.annotations_dict=load_volleyball_dataset(videos_root,annot_root)
        # convert the dict to list of clips
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
    
    # we work in image level, for each clip we need middle frame with clip label 
    def __getitem__(self, index):
        item=self.samples[index]
        middle_frame_key=list(item['frame_boxes_dct'].keys())[4] # 9 frames -> middle = index 4
        middle_frame_path=os.path.join(self.videos_root,item['video_id'],item['clip_id'],f'{middle_frame_key}.jpg')
        
        # JPEG files are not all created equal. depending on how they were saved, .jpg can be: RGB, grayscale
        # By calling .convert('RGB'), we enforce a shape invariant. it ensures that every single tensor that enters your model has exactly 3 channels
        img=Image.open(middle_frame_path).convert('RGB')
        img=self.preprocess(img) #return tensor 

        label=torch.tensor(self.team_action_dct[item['category']])

        return img,label
    
 
# annot_root=r"D:\track\Deep learning\cskill\slides\05 Volleyball Project\sample data\volleyball_tracking_annotation"
# videos_root=r"D:\track\Deep learning\cskill\slides\05 Volleyball Project\videos_g10"

# img_da=VolleyballImageDataset(videos_root,annot_root)

# print(img_da.__len__())
# img,label=img_da.__getitem__(10)
# print(img.shape)
# print(label.shape) 