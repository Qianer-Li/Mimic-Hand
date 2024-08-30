import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
import cv2
from controlnet_aux.util import HWC3, resize_image

import decord
import numpy as np
import torch.jit 
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader 
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image
import imageio
from PIL import Image

from constants import ASPECT_RATIO

from mimicmotion.pipelines.pipeline_mimicmotion2 import MimicMotionPipeline2
from mimicmotion.utils.loader import create_pipeline2  
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose  

from decord import VideoReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(video_path, image_path, resolution=576, sample_stride=2):
    """preprocess ref image pose and video pose  视频和图像预处理函数

    Args:
        video_path (str): input video pose path 输入视频路径
        image_path (str): reference image path 参考图像路径
        resolution (int, optional):  Defaults to 576.  输出分辨率，默认为576
        sample_stride (int, optional): Defaults to 2.  采样步长，默认为2
    """
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    ref_image  = image_pixels
    h, w = image_pixels.shape[-2:]

    if h>w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
        
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
        
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    
    # 获取图像和视频的姿态数据
    image_pose = get_image_pose(image_pixels)
    video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    # print(image_pixels.shape)
    return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1, image_pixels

def preprocess_hand(hand_path, hand_mask_path, images_depth_path, images_mask_path, image,
                    resolution=576, sample_stride=2, img_scale=(1.0, 1.0), 
                    img_ratio=(0.9, 1.0), drop_ratio=0.1,):
    # print(image_pixels.shape)
    height, width = image.shape[-2], image.shape[-1]
    
    image_depth = Image.fromarray(np.array(pil_loader(images_depth_path)))
    images_mask = Image.fromarray(np.array(pil_loader(images_mask_path)))
        
    hand_transform = transforms.Compose([
        transforms.RandomResizedCrop((height, width), scale=img_scale, 
        ratio=img_ratio, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    hand_mask_transform = transforms.Compose([
        transforms.RandomResizedCrop((height, width), scale=img_scale, 
        ratio=img_ratio, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    
    image_depth = torch.tensor(np.array(hand_transform(image_depth))).unsqueeze(0)
    images_mask = torch.tensor(np.array(hand_mask_transform(images_mask))).unsqueeze(0)
    print(image_depth.shape, images_mask.shape)
    
    hand_reader = VideoReader(hand_path)
    hand_mask_reader = VideoReader(hand_mask_path)
    
    sample_stride *= max(1, int(hand_reader.get_avg_fps() / 24))
    
    detected_hand_poses = [
        frm for frm in hand_reader.get_batch(
            list(range(0, len(hand_reader), sample_stride))
        ).asnumpy()
    ]
    
    detected_hand_masks = [
        frm for frm in hand_mask_reader.get_batch(
            list(range(0, len(hand_mask_reader), sample_stride))
        ).asnumpy()
    ]
    
    # 应用图像变换
    hand_pil_image_list = [Image.fromarray(frm) for frm in detected_hand_poses]
    hand_mask_pil_image_list = [Image.fromarray(frm) for frm in detected_hand_masks]
    
    # 转换后的手部图像和遮罩图像
    transformed_hand_images = [hand_transform(img) for img in hand_pil_image_list]
    transformed_hand_masks = [hand_mask_transform(mask) for mask in hand_mask_pil_image_list]
    
    # 将图像和遮罩图像转换为tensor
    transformed_hand_images = [torch.tensor(np.array(img)) for img in transformed_hand_images]
    transformed_hand_masks = [torch.tensor(np.array(mask)) for mask in transformed_hand_masks]
    
    # 堆叠图像列表以获得输出张量
    output_hand_images = torch.stack(transformed_hand_images)
    output_hand_masks = torch.stack(transformed_hand_masks)
    print(output_hand_images.shape, output_hand_masks.shape)
    
    # 在0维增加一度
    output_hand_images = torch.cat((image_depth, output_hand_images), dim=0) 
    output_hand_masks = torch.cat((images_mask, output_hand_masks), dim=0) 
    
    output_hand_images = output_hand_images.unsqueeze(0).permute(0, 2, 1, 3, 4)
    output_hand_masks = output_hand_masks.unsqueeze(0).permute(0, 2, 1, 3, 4)

    return output_hand_images, output_hand_masks

def run_pipeline(pipeline: MimicMotionPipeline2, image_pixels, pose_pixels, hand_images, hand_masks, device, task_config, flag=False):
    """运行模拟动作管道resize_image

    Args:
        pipeline (MimicMotionPipeline): 模拟动作管道实例
        image_pixels (torch.Tensor): 图像像素数据
        pose_pixels (torch.Tensor): 姿势像素数据
        device (torch.device): 运行设备
        task_config (OmegaConf): 任务配置
    """
    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    pose_pixels = pose_pixels.unsqueeze(0).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    
    frames = pipeline(
        image_pixels, image_pose=pose_pixels, hand_images=hand_images, 
        hand_masks=hand_masks,num_frames=pose_pixels.size(1),
        tile_size=task_config.num_frames, tile_overlap=task_config.frames_overlap,
        height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
        noise_aug_strength=task_config.noise_aug_strength, num_inference_steps=task_config.num_inference_steps,
        generator=generator, min_guidance_scale=task_config.guidance_scale, 
        max_guidance_scale=task_config.guidance_scale, decode_chunk_size=8, output_type="pt", device=device
    ).frames.cpu()
    video_frames = (frames * 255.0).to(torch.uint8)
    
    for vid_idx in range(video_frames.shape[0]):
        # deprecated first frame because of ref image
        _video_frames = video_frames[vid_idx, 1:]
        if vid_idx == 10:
            break
    return _video_frames


@torch.no_grad()
def main(args):
    if not args.no_use_float16 :
        torch.set_default_dtype(torch.float16)
    
    infer_config = OmegaConf.load(args.inference_config) 
    pipeline = create_pipeline2(infer_config, device)  # TODO: 创建并加载模型
    
    save_dir = "../autodl-tmp/frames"
    for task in infer_config.test_case:
        
        task_output_dir = os.path.join(save_dir, f"{os.path.basename(task.ref_video_path).split('.')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        os.makedirs(task_output_dir, exist_ok=True)
        
        pose_pixels, image_pixels, image = preprocess(
            task.ref_video_path, task.ref_image_path, 
            resolution=task.resolution, sample_stride=task.sample_stride
        )
        
        hand_images, hand_masks = preprocess_hand(
            task.ref_hand_path, task.ref_hand_mask_path, 
            task.images_depth_path, task.images_mask_path,
            image, sample_stride=task.sample_stride
        )
        
        print('image_pixels.shape, pose_pixels.shape', image_pixels.shape, pose_pixels.shape)
        print('hand_images.shape, hand_masks.shape', hand_images.shape, hand_masks.shape)
        
        _video_frames = run_pipeline(
            pipeline, 
            image_pixels, pose_pixels, 
            hand_images, hand_masks, 
            device, task
        )
        
#         frames_save_dir = os.path.join(task_output_dir, 'frames')
#         os.makedirs(frames_save_dir, exist_ok=True)
#         for i, frame in enumerate(_video_frames):
#             input_image = cv2.cvtColor(
#                 np.array(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR
#             )
#             input_image = HWC3(input_image)
#             # input_image = resize_image(input_image, detect_resolution)
#             H, W, C = input_image.shape
#             print(input_image.shape)

#             frame_save_path = os.path.join(frames_save_dir, f"frame_{i+1:04d}.png")
#             cv2.imwrite(frame_save_path, input_image)
            
        save_to_mp4(
            _video_frames, 
            f"{args.output_dir}/{os.path.basename(task.ref_video_path).split('.')[0]}" \
            f"_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4",
            fps=task.fps,
        )

def set_logger(log_file=None, log_level=logging.INFO):
    log_handler = logging.FileHandler(log_file, "w")
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    )
    log_handler.setLevel(log_level)
    logger.addHandler(log_handler)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--inference_config", type=str, default="configs/test.yaml") #ToDo
    parser.add_argument("--output_dir", type=str, default="outputs/", help="path to output")
    parser.add_argument("--no_use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_logger(args.log_file \
               if args.log_file is not None else f"{args.output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    main(args)
    logger.info(f"--- Finished ---")

