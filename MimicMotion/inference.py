import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.jit  # 提供模型编译功能，用于优化执行
from torchvision.datasets.folder import pil_loader  # 用于加载图像文件为PIL格式
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image

from constants import ASPECT_RATIO

from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline  # 导入模拟动作管道
from mimicmotion.utils.loader import create_pipeline  # 用于创建处理管道
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose  # 导入视频和图像姿势处理函数

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
    h, w = image_pixels.shape[-2:]

    # 计算目标高宽比
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
    return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1


def run_pipeline(pipeline: MimicMotionPipeline, image_pixels, pose_pixels, device, task_config):
    """运行模拟动作管道

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
        image_pixels, image_pose=pose_pixels, num_frames=pose_pixels.size(1),
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
        
    return _video_frames


@torch.no_grad()
def main(args):
    if not args.no_use_float16 :
        torch.set_default_dtype(torch.float16)

    infer_config = OmegaConf.load(args.inference_config)  # 加载推理配置文件
    pipeline = create_pipeline(infer_config, device)  # TODO: 创建并加载模型

    for task in infer_config.test_case:
         # 预处理数据
        pose_pixels, image_pixels = preprocess(
            task.ref_video_path, task.ref_image_path, 
            resolution=task.resolution, sample_stride=task.sample_stride
        )
        
        print('image_pixels pose_pixels shape', image_pixels.shape, pose_pixels.shape)
        # 运行模拟动作管道
        _video_frames = run_pipeline(
            pipeline, 
            image_pixels, pose_pixels, 
            device, task
        )
        print('_video_frames', _video_frames.shape)
        
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

