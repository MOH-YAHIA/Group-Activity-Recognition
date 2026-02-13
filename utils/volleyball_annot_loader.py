import os
import cv2
from utils.boxinfo import BoxInfo
import torch
import logging

logger = logging.getLogger(__name__)

def load_tracking_annot(path):
    '''
        get the annotation for the players in all frames in this clip
        path: clip annotaion path
    '''
    with open(path, 'r') as file:
        player_boxes = {idx:[] for idx in range(12)} #12 players
        frame_boxes_dct = {}
        for idx, line in enumerate(file):
            box_info = BoxInfo(line)
            # may be more than 12 player in the clip -> ignore
            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)
        # let's create view from frame to boxes
        for player_ID, boxes_info in player_boxes.items():
            # each player has 20 frames sorted according to frame_ID
            # let's keep the middle 9 frames only (enough for this task empirically)
            
            boxes_info = boxes_info[5:]
            boxes_info = boxes_info[:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []

                frame_boxes_dct[box_info.frame_ID].append(box_info)

        #dic contains boxes info for players in each frame
        #9 frames, each contains boxes info for 12 players 
        return frame_boxes_dct
    

def vis_clip(annot_path, video_dir):
    frame_boxes_dct = load_tracking_annot(annot_path)
    font = cv2.FONT_HERSHEY_SIMPLEX # text font 

    for frame_id, boxes_info in frame_boxes_dct.items():
        img_path = os.path.join(video_dir, f'{frame_id}.jpg')
        image = cv2.imread(img_path)

        for box_info in boxes_info:
            x1, y1, x2, y2 = box_info.box
            #                                         colur     thickness
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, box_info.category, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

        cv2.imshow('Image', image)
        cv2.waitKey(180)
    cv2.destroyAllWindows()


def load_video_annot(video_annot):
    '''
        get the category for each clip in video dir
    '''
    with open(video_annot, 'r') as file:
        clip_category_dct = {}
        for line in file:
            # line looks like: "12345.jpg r_set" (filename and action_id)
            items = line.strip().split(' ')[:2]
            
            # Removes '.jpg' to get just the clip folder name (e.g., "12345")
            clip_dir = items[0].replace('.jpg', '')
            
            # Maps the directory name to its action category
            clip_category_dct[clip_dir] = items[1]

        return clip_category_dct


def load_volleyball_dataset(videos_root, annot_root):
    # videos_root -> videos
    # annot_root -> volleyball_tracking_annotation

    # check for preloaded annotations
    annot_path=os.path.join('data','video_annot.pth')
    if os.path.exists(annot_path):
        logger.info(f"Loading cached annotations from {annot_path}")
        videos_annot=torch.load(annot_path,weights_only=False)
        return videos_annot
    
    videos_dirs = os.listdir(videos_root) #get folders and files names '0','1','2','readme.txt'
    videos_dirs.sort()

    videos_annot = {}

    # Iterate on each video and for each video iterate on each clip
    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path): #in case there is file not folder as 'readme.txt'
            logger.debug("Skipping non-dir: %s", video_dir) # Low-level noise
            continue

        logger.info(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        video_annot_path = os.path.join(video_dir_path, 'annotations.txt')
        clip_category_dct = load_video_annot(video_annot_path)

        clips_dir = os.listdir(video_dir_path) #clibs id's
        clips_dir.sort()

        clip_annot = {}

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            #logger.info(f'\t{clip_dir_path}')
            assert clip_dir in clip_category_dct

            annot_file = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
            frame_boxes_dct = load_tracking_annot(annot_file)
            #vis_clip(annot_file, clip_dir_path)

            clip_annot[clip_dir] = {
                'category': clip_category_dct[clip_dir],
                'frame_boxes_dct': frame_boxes_dct #9 frames 
            }

        videos_annot[video_dir] = clip_annot
    if not os.path.exists('data'):
        os.mkdir('data')
    logger.info(f"Saving processed annotations to {annot_path}")
    torch.save(videos_annot,annot_path)
    return videos_annot

'''
videos_annot look like
{
  "0": {                  # Video ID
    "13456": {            # Clip ID
      "category": "r_set",    # Action Label
      "frame_boxes_dct": {
         13454: [BoxInfo, BoxInfo, ...], #frame_ID: 12 Players
         13455: [BoxInfo, BoxInfo, ...],
         ... # 9 frames total
      }
    }
  }
}
'''



# for video_id,clips in videos_annot.items():
#     logger.info(video_id)
#     for clip_id,clip in clips.items():
#         logger.info("--",clip_id)
#         logger.info('--',clip['category'])
#         for frame_id,players_boxs in clip['frame_boxes_dct'].items():
#             logger.info('------',frame_id, len(players_boxs)) 
