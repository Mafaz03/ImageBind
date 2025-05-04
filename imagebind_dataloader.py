import torch
from torch import nn
from config import IMAGE_TRANSFORM, VIDEO_TRANSFORM
from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image
import os
from preprocessing import uniform_crop, Spacial_Crop, get_clip_timepoints, load_and_transform_text
import matplotlib.pyplot as plt
import clip

from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.encoded_video import EncodedVideo
import av

class ImageText_DataLoader(Dataset):
    def __init__(self, image_paths, transform, device = "cpu"):
        super().__init__()
        
        if image_paths is None: return None
        
        self.image_outputs = []
        self.text_outputs = []

        for image_path in os.listdir(image_paths):
            if image_path.endswith(".jpg") or image_path.endswith(".jpeg") or image_path.endswith(".png"):
                with open(os.path.join(image_paths, image_path), "rb") as im:
                    image = Image.open(im).convert("RGB")
                    image = transform(img = image)
                    self.image_outputs.append(image)
                    self.text_outputs.append(image_path.split('.')[0])
        self.text_outputs = load_and_transform_text(self.text_outputs)
        # return torch.stack(image_outputs, dim=0)

    def __len__(self):
        return len(self.image_outputs) or len(self.text_outputs)
    
    def __getitem__(self, index):
        return (self.image_outputs[index], self.text_outputs[index])


class VideoText_DataLoader(Dataset):
    def __init__(self, video_paths: str, transform, clip_duration=2, clips_per_video=5, device = "cpu"):
        super().__init__()

        self.video_outputs = []
        self.text_outputs = []
        
        video_transform = transform

        clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video, 
        )

        frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=2)
        for video_path in os.listdir(video_paths): 
            if video_path.endswith(".mp4"):
                encoded_video = EncodedVideo.from_path(
                    file_path=f"Data/Video_Data/{video_path}",
                    decode_audio=False,
                    decoder="pyav"
                )

                all_frames = []

                clip_time_points = get_clip_timepoints(clip_sampler=clip_sampler, duration=encoded_video.duration)

                for clip_time_point in clip_time_points:
                    clip = encoded_video.get_clip(clip_time_point[0], clip_time_point[1])
                    if clip is None: ValueError("No Clip Found") 
                    frames = frame_sampler(clip["video"]) / 255.0

                    all_frames.append(frames)
                all_videos = [video_transform(frame) for frame in all_frames]
                self.video_outputs.append(Spacial_Crop(crop_size = 224, num_crops = 3)(all_videos))
                self.text_outputs.append(video_path.split('.')[0])
        self.text_outputs = load_and_transform_text(self.text_outputs)

    def __len__(self):
        return len(self.video_outputs)
    
    def __getitem__(self, index):
        return self.video_outputs[index], self.text_outputs[index]

