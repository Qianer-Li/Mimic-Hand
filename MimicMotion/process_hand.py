# STEP 1: Import the necessary modules.
from __future__ import absolute_import, division, print_function
import sys
from config import mimicmotion_root
import os

def load():
    paths = [mimicmotion_root, os.path.join(mimicmotion_root, 'MeshGraphormer'),\
             os.path.join(mimicmotion_root, 'mimicmotion', 'modules'),\
             os.path.join(mimicmotion_root, 'dataset')]
    for p in paths:
        sys.path.insert(0, p)
load()

import argparse
import json
import torch
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

from pytorch_lightning import seed_everything
import config

import cv2
import ast
import einops
import numpy as np
import torch
import random
from pathlib import Path

import mediapipe as mp
# from mediapipe.tasks.python.vision import ImageFormat
from controlnet_aux.util import HWC3, resize_image
from mimicmotion.modules.meshgraphormer import MeshGraphormerMediapipe
from mimicmotion.dwpose.util import draw_bodypose, draw_handpose, draw_facepose
from mimicmotion.utils.utils import get_fps, save_videos_from_pil, read_frames, read_handframes

meshgraphormer = MeshGraphormerMediapipe()

def convert_to_mp_image(frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    return mp_image

def save_mp_image_to_path(mp_image, file_path):
    # Convert mp.Image to a NumPy array
    image_np = mp_image.numpy_view()

    # Convert RGB to BGR for OpenCV (if using OpenCV)
    # OpenCV expects images in BGR format
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Save the image using OpenCV
    cv2.imwrite(file_path, image_bgr)

    # Alternatively, save using PIL
    # Convert NumPy array to PIL image
    image_pil = PILImage.fromarray(image_np)

    # Save the image using PIL
    image_pil.save(file_path)
    
def draw_hand(pose, H, W):
    bodies = pose["bodies"]
    
    faces = pose["faces"]
    faces_score = pose["faces_score"]
    hands = pose["hands"]
    hands_score = pose["hands_score"]
    
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    score = bodies["score"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    
    canvas = draw_bodypose(canvas, candidate, subset, score)
    
    canvas = draw_handpose(canvas, hands, hands_score)

    canvas = draw_facepose(canvas, faces, faces_score)

    return canvas

def process_single_video_hand(video_path, detector, root_dir, save_dir, save_mask_dir, detect_resolution=512, image_resolution=512, output_type="pil", padding_bbox=30):
    relative_path = os.path.relpath(video_path, root_dir)
    
    out_path = os.path.join(save_dir, relative_path)
    out_mask_path = os.path.join(save_mask_dir, relative_path)
    print('relative_path, video_path, root_dir', relative_path, video_path, root_dir)
    if os.path.exists(out_path) and os.path.exists(out_mask_path) :
        return
    
    output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    output_mask_dir = Path(os.path.dirname(os.path.join(save_mask_dir, relative_path)))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        
    if not output_mask_dir.exists():
        output_mask_dir.mkdir(parents=True, exist_ok=True)
        
    fps = get_fps(video_path)
    frames = read_frames(video_path)
    kps_results = []
    kps_mask_results = []
    for i, frame_pil in enumerate(frames):
        print(f"Processing frame {i+1}", end='\r')
        
        input_image = cv2.cvtColor(
            np.array(frame_pil, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        
        # input_image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_image)
        # input_image = convert_to_mp_image(frame_pil)
        depthmap, mask, info = detector.get_hand(input_image, padding_bbox, relative_path)
        
        if depthmap is None:
            depthmap = np.zeros((H, W, 3), dtype=np.uint8)
            depthmap = HWC3(depthmap)
            mask = np.zeros((H, W, 3), dtype=np.uint8)
            mask = HWC3(mask)
        else:
            depthmap = HWC3(depthmap)
            mask = HWC3(mask)

        depthmap = resize_image(depthmap, image_resolution)
        H, W, C = depthmap.shape
        depthmap = cv2.resize(depthmap, (W, H), interpolation=cv2.INTER_LINEAR)
        
        mask = resize_image(mask, image_resolution)
        H, W, C = mask.shape
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
        
        if output_type == "pil":
            depthmap = Image.fromarray(depthmap)
            mask = Image.fromarray(mask)
            
        kps_results.append(depthmap)
        kps_mask_results.append(mask)
        
    save_videos_from_pil(kps_results, out_path, fps=fps)
    save_videos_from_pil(kps_mask_results, out_mask_path, fps=fps)

def process_single_video_hands(video_path, detector, root_dir, save_dir, save_mask_dir, detect_resolution=512, image_resolution=512, output_type="pil", padding_bbox=30):
    relative_path = os.path.relpath(video_path, root_dir)
    
    out_path = os.path.join(save_dir, relative_path)
    out_mask_path = os.path.join(save_mask_dir, relative_path)
    print('relative_path, video_path, root_dir', relative_path, video_path, root_dir)
    if os.path.exists(out_path) and os.path.exists(out_mask_path) :
        return
    
    output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    output_mask_dir = Path(os.path.dirname(os.path.join(save_mask_dir, relative_path)))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        
    if not output_mask_dir.exists():
        output_mask_dir.mkdir(parents=True, exist_ok=True)
        
    fps = get_fps(video_path)
    frames = read_frames(video_path)
    kps_results = []
    kps_mask_results = []
    for i, frame_pil in enumerate(frames):
        print(f"Processing frame {i+1}", end='\r')
        
        input_image = cv2.cvtColor(
            np.array(frame_pil, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        
        input_image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_image)
        # input_image = convert_to_mp_image(frame_pil)
        depthmap, mask, info = detector.get_hands(input_image_mp, padding_bbox, relative_path)
        
        if depthmap is None:
            depthmap = np.zeros((H, W, 3), dtype=np.uint8)
            depthmap = HWC3(depthmap)
            mask = np.zeros((H, W, 3), dtype=np.uint8)
            mask = HWC3(mask)
        else:
            depthmap = HWC3(depthmap)
            mask = HWC3(mask)
        # kps_results.append(depthmap)
        # frame_save_path = os.path.join(output_dir, f"frame_{i+1:04d}.png")
        # cv2.imwrite(frame_save_path, input_image)
        
        # detected_map = draw_pose(pose, H, W)
        
        depthmap = resize_image(depthmap, image_resolution)
        H, W, C = depthmap.shape
        depthmap = cv2.resize(depthmap, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # frame_save_path = os.path.join(output_dir, f"frame_{i+1:04d}.png")
        # cv2.imwrite(frame_save_path, input_image)
        
        mask = resize_image(mask, image_resolution)
        H, W, C = mask.shape
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
        
        if output_type == "pil":
            depthmap = Image.fromarray(depthmap)
            mask = Image.fromarray(mask)
        
        kps_results.append(depthmap)
        kps_mask_results.append(mask)
        
    save_videos_from_pil(kps_results, out_path, fps=fps)
    save_videos_from_pil(kps_mask_results, out_mask_path, fps=fps)
    
def process_batch_videos_hands(video_list, detector, root_dir, save_dir, pose_mask_dir):
    for i, video_path in enumerate(video_list):
        print(f"Process {i}/{len(video_list)} video")
        process_single_video_hands(video_path, detector, root_dir, save_dir, pose_mask_dir)
        
def process_batch_videos_hand(video_list, detector, root_dir, save_dir, pose_mask_dir):
    for i, video_path in enumerate(video_list):
        print(f"Process {i}/{len(video_list)} video")
        process_single_video_hand(video_path, detector, root_dir, save_dir, pose_mask_dir)


if __name__ == "__main__":
    pose_root_dir = "../datas/TikTok_Crop_Videos"
    pose_save_dir = pose_root_dir + "_dwhand"
    pose_mask_dir = pose_root_dir + "_dwmask"

    pose_mp4_paths = set()
    for root, dirs, files in os.walk(pose_root_dir):
        for name in files:
            if name.endswith(".mp4"):
                pose_mp4_paths.add(os.path.join(root, name))
    pose_mp4_paths = list(pose_mp4_paths)
    
    dtype="hand"
    if dtype=="hands":
        process_batch_videos_hands(pose_mp4_paths, meshgraphormer, root_dir=pose_root_dir, save_dir=pose_save_dir,\
                                   pose_mask_dir=pose_mask_dir)
    elif dtype=="hand":
        process_batch_videos_hand(pose_mp4_paths, meshgraphormer, root_dir=pose_root_dir, save_dir=pose_save_dir,\
                                 pose_mask_dir=pose_mask_dir)

# from mimicmotion.modules.control_net import ControlNet