from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np
from config import IMAGE_TRANSFORM
import torch
import clip
import os
import math
from pytorchvideo import transforms as pv_transforms
from torchvision.transforms._transforms_video import NormalizeVideo
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
import av

def load_and_transform_vision_date(image_paths, device="cpu"):
    if image_paths is None: return None
    image_outputs = []

    for image_path in os.listdir(image_paths):
        if image_path.endswith(".jpg") or image_path.endswith(".jpeg") or image_path.endswith(".png"):
            with open(os.path.join(image_paths, image_path), "rb") as im:
                image = Image.open(im).convert("RGB")
                image = IMAGE_TRANSFORM(img = image)
                image_outputs.append(image)
    return torch.stack(image_outputs, dim=0)

def load_and_transform_text(texts, device="cpu"):
    if texts is None: return None
    tokens = [clip.tokenize(text).unsqueeze(0).to(device) for text in texts]
    return torch.cat(tokens, dim=0)

def uniform_crop(images, size, spacial_idx, boxes=None, scale_size=None):
    assert spacial_idx in [0, 1, 2], "`spacial_idx` must be 0 -> Left | 1 -> Center | 2 -> Right"
    height = images.shape[2]
    width = images.shape[3]
    
    if scale_size: # Scale size keeps the ratio of original height and widht intact but changes the shorter size to `scale_size`
        if height <= width:
            width, height = int((width/height)*scale_size), scale_size
        else:
            width, height = scale_size, int((height/width)*scale_size)
        
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode = "bilinear",
            align_corners=False
        )
    y_offset = math.ceil((height-size)/2)
    x_offset = math.ceil((width-size)/2)

    if height <= width:
        if spacial_idx == 0:
            x_offset = 0
        elif spacial_idx == 2:
            x_offset = width - size
    else:
        if spacial_idx == 0:
            y_offset = 0
        elif spacial_idx == 2:
            y_offset = height - size

    return images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]

class Spacial_Crop(nn.Module):
    def __init__(self, crop_size: int, num_crops: int = 3):
        super().__init__()

        self.crop_size = crop_size
        if num_crops == 3:
            self.crop_directions = [0, 1, 2] # 0 -> Left | 1 -> Center | 2 -> Right
        elif num_crops == 1:
            self.crop_directions = [1] # 1 -> Center
        else:
            raise NotImplementedError("Make sure the `num_crops` is either 1 or 3, 3 being default")
    
    def forward(self, videos):
        res = []
        for video in videos:
            for spacial_idx in self.crop_directions:
                res.append(uniform_crop(video, size=self.crop_size, spacial_idx=spacial_idx))
        return torch.stack(res, dim=0)
    
def get_clip_timepoints(clip_sampler, duration):
    is_last_clip = False
    end = 0.9
    clip_time_points = []
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(0.0, duration, annotation=False)
        clip_time_points.append((start, end))
    return clip_time_points

def load_and_transform_video_data(video_paths, device, clip_duration=2, clips_per_video=5):
    video_outputs = []
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

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
            video_outputs.append(Spacial_Crop(crop_size = 224, num_crops = 3)(all_videos))
    return torch.stack(video_outputs, dim=0).to(device)
    