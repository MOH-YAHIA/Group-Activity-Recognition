from torch.utils.data import Dataset
from utils.volleyball_annot_loader import load_volleyball_dataset

class BaseDataset(Dataset):
    def __init__(self,videos_root,annot_root,allowed_ids,one_frame):
        self.videos_root=videos_root
        self.one_frame=one_frame
        self.team_action_dct = {
        'l-pass': 0,     'r-pass': 1,
        'l-spike': 2,    'r_spike': 3,
        'l_set': 4,      'r_set': 5,
        'l_winpoint': 6, 'r_winpoint': 7
        }
        # get the annotation for the whole dataset in a dictionary
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
    

    
        