

import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
import numpy as np
import gradio as gr

import torch

from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
from inference import preprocess, run_pipeline



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------------------------

def run_MimicMotion(
	ref_image_path,
	ref_video_path,
	num_frames,
	resolution,
	frames_overlap,
	num_inference_steps,
	noise_aug_strength,
	guidance_scale,
	sample_stride,
	fps,
	seed,
	use_fp16,
):
	if use_fp16:
		torch.set_default_dtype(torch.float16)
	
	infer_config = OmegaConf.create({
		'base_model_path': 'models/SVD/stable-video-diffusion-img2vid-xt-1-1',
		'ckpt_path': 'models/MimicMotion.pth',
		'test_case': [
            {
                'ref_video_path': ref_video_path,
                'ref_image_path': ref_image_path,
                'num_frames': num_frames,
                'resolution': resolution,
                'frames_overlap': frames_overlap,
                'num_inference_steps': num_inference_steps,
                'noise_aug_strength': noise_aug_strength,
                'guidance_scale': guidance_scale,
                'sample_stride': sample_stride,
                'fps': fps,
                'seed': seed,
            },
        ],
	})

	pipeline = create_pipeline(infer_config, device)

	for task in infer_config.test_case:
		# Pre-process data
		pose_pixels, image_pixels = preprocess(
			task.ref_video_path, task.ref_image_path, 
			resolution=task.resolution, sample_stride=task.sample_stride
		)

		# Run MimicMotion pipeline
		_video_frames = run_pipeline(
			pipeline, 
			image_pixels, pose_pixels, 
			device, task
		)

		################################### save results to output folder. ###########################################
		now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		output_dir = os.path.dirname(os.path.abspath(__file__))
		output_dir = os.path.join(output_dir, 'outputs')
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)
		filename = os.path.splitext(os.path.basename(task.ref_image_path))[0]
		path_out_vid = os.path.join(output_dir, f'{filename}_{now_str}.mp4')
		print(f'Video will be saved to: {path_out_vid}')
		save_to_mp4(_video_frames, path_out_vid, fps=task.fps)
		print('OK !')

	return path_out_vid

#-----------------------------------------------------------------------

with gr.Blocks() as demo:
	with gr.Row():
		gr.Markdown("""
			<h2><a href="https://github.com/Tencent/MimicMotion" target="_blank" style="text-decoration: none">MimicMotion</a>：利用置信度感知姿势引导生成高质量人体运动视频</h2>
			<h4>gradio demo 由 <a href="https://space.bilibili.com/1872960954/video" target="_blank" style="text-decoration: none">为了它O_O</a> 制作</h4>
		""")
	with gr.Row():
		with gr.Column():
			gr_ref_img = gr.Image(label='参考图片', type='filepath')
			gr_ref_vid = gr.Video(label='参考视频')
		with gr.Column():
			gr_out_vid = gr.Video(label='生成结果', interactive=False)
			with gr.Accordion(label='参数设置'):
				gr_num_frames = gr.Number(label='总帧数', value=16)
				gr_resolution = gr.Number(label='分辨率', value=576)
				gr_frames_overlap = gr.Number(label='重叠帧数', value=6)
				gr_infer_steps = gr.Number(label='推理步数', value=25)
				gr_noise_aug_strength = gr.Number(label='噪声强度', value=0.0)
				gr_guidance_scale = gr.Number(label='引导系数', value=2.0)
				gr_sample_stride = gr.Number(label='采样步长', value=2)
				gr_fps = gr.Number(label='帧率', value=15)
				gr_seed = gr.Number(label='种子', value=42)
				gr_use_fp16 = gr.Checkbox(label='使用float16', value=True)
			gr_btn = gr.Button(value='生成视频')
	
	gr_btn.click(
		fn=run_MimicMotion,
		inputs=[
			gr_ref_img,
			gr_ref_vid,
			gr_num_frames,
			gr_resolution,
			gr_frames_overlap,
			gr_infer_steps,
			gr_noise_aug_strength,
			gr_guidance_scale,
			gr_sample_stride,
			gr_fps,
			gr_seed,
			gr_use_fp16,
		],
		outputs=gr_out_vid,
	)
			

demo.launch()


