import sys
import os
import os.path as osp
from pathlib import Path
mimicmotion_root=str(Path(__file__).parent)

def load():
    paths = [mimicmotion_root, os.path.join(mimicmotion_root, 'MeshGraphormer'),\
             os.path.join(mimicmotion_root, 'mimicmotion', 'modules'),\
             os.path.join(mimicmotion_root, 'mimicmotion', 'dataset')]
    for p in paths:
        sys.path.insert(0, p)
load()

import argparse
import copy
import logging
import math
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from tempfile import TemporaryDirectory
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import copy

import einops
import diffusers
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import transformers
from transformers import CLIPVisionModelWithProjection
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline 
from mimicmotion.pipelines.pipeline_mimicmotion2 import MimicMotionPipeline2 
from mimicmotion.utils.loader import MimicMotionModel, MimicMotionModel2  
from mimicmotion.modules.pose_net import PoseNet
from mimicmotion.modules.control_net import ControlNet
from mimicmotion.modules.meshgraphormer import MeshGraphormerMediapipe
from mimicmotion.utils.utils import save_to_mp4
# from inference import preprocess, run_pipeline
from inference2 import preprocess, preprocess_hand, run_pipeline


from dataset.dance_video import HumanDanceVideoDataset, MimicMotionDataset

from mimicmotion.utils.utils import (
    delete_additional_ckpt,
    import_filename,
    read_frames,
    save_videos_grid,
    seed_everything,
)
from mimicmotion.modules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from mimicmotion.modules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from mimicmotion.modules.encoder import FrozenCLIPEmbedder
from mimicmotion.modules.control_net import ControlNet
torch.autograd.set_detect_anomaly(True)

warnings.filterwarnings("ignore")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# python train_stage_4.py
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def clone_tensors(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        return tensor_list.clone()
    elif isinstance(tensor_list, list):
        return [clone_tensors(t) for t in tensor_list]
    else:
        raise TypeError("Expected input to be a list or torch.Tensor")
        
class Net(nn.Module):
    def __init__(
        self,
        unet: UNetSpatioTemporalConditionModel,
        pose_net: PoseNet,
        control_net: ControlNet,
    ):
        super().__init__()
        self.unet = unet
        self.pose_guider = pose_net
        self.control_net =control_net
        self.control_scales = [1.0] * 13
        self.control_weight = 0.2
    
    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pixel_values_pose,
        cond,
        added_time_ids,
        uncond_fwd: bool = False,
        tile_size = 4,
        tile_overlap = 1,
        image_only_indicator: bool = False,
        global_average_pooling = False,
    ):
        num_frames = noisy_latents.shape[2]
        
        hint = torch.cat(cond['c_control'], 1).to(device=device)
        
        # cond['c_control(深度图)
        # print(f"pose_latents shape x_noisy: {noisy_latents.shape}")
        # print(f"pose_latents shape hint: {torch.cat(cond['c_control'], 1).shape}")
        # print(f"pose_latents shape t: {timesteps.shape}")
        # print(f"pose_latents shape cond_txt: {cond_txt.shape}")
        
        # TODO: Origion
        pose_cond_tensor = pixel_values_pose.permute(0, 2, 1, 3, 4).to(device=device)
        noisy_latents = noisy_latents.permute(0, 2, 1, 3, 4).to(device=device)
        ref_image_latents = ref_image_latents.permute(0, 2, 1, 3, 4).to(device=device)
        hint_cond_tensor = hint.permute(0, 2, 1, 3, 4).to(device=device)
        
        pose_latents = self.pose_guider(pose_cond_tensor)
        # print(f"pose_latents shape after pose_guider: {pose_latents.shape}")
        
        hint_latents = self.control_net(hint_cond_tensor)
        # print(f"hint_cond_tensor shape after: {hint_latents.shape}")
        
        indices = [[0, *range(i + 1, min(i + tile_size, num_frames))] for i in
                   range(0, num_frames - tile_size + 1, tile_size - tile_overlap)]
        if indices[-1][-1] < num_frames - 1:
            indices.append([0, *range(num_frames - tile_size + 1, num_frames)])
        
        latent_model_input = torch.cat([noisy_latents, ref_image_latents], dim=2)
        pose_latents = einops.rearrange(pose_latents, '(b f) c h w -> b f c h w', f=num_frames)
        hint_latents = einops.rearrange(hint_latents, '(b f) c h w -> b f c h w', f=num_frames)
        # print(f"Shape of rearranged pose_latents: {pose_latents.shape, hint_latents.shape}")
        
        # predict the noise residual
        noise_pred = torch.zeros_like(ref_image_latents)
        noise_pred_cnt = ref_image_latents.new_zeros((num_frames,))
        
        weight = (torch.arange(tile_size, device=device) + 0.5) * 2. / tile_size
        weight = torch.minimum(weight, 2 - weight)
        
        for idx in indices:
            # print(f"Shape of latent_model_input[:, idx]: {latent_model_input[:, idx].shape}")
            # print(f"Shape of t: {timesteps.shape}")
            # print(f"Shape of encoder_hidden_states (clip_image_embeds): {clip_image_embeds.shape}")
            # print(f"Shape of added_time_ids: {added_time_ids.shape}")
            # print(f"Shape of pose_latents[:, idx]: {pose_latents[:, idx].shape}")
            
            idx_tensor = torch.tensor(idx, device=noisy_latents.device) 
            assert idx_tensor.max() < noise_pred.size(1), f"Index {idx_tensor.max()} out of bounds\
            for dimension 1 of noise_pred with size {noise_pred.size(1)}"
            
            _noise_pred = self.unet(
                latent_model_input[:, idx],
                timesteps,
                encoder_hidden_states=clip_image_embeds,
                added_time_ids=added_time_ids,
                pose_latents=pose_latents[:, idx].flatten(0, 1),  # 展平向量
                return_dict=False,
                image_only_indicator = image_only_indicator,
                control_latents = hint_latents[:, idx].flatten(0, 1)*self.control_weight,
            )[0]
            
            # print(f"Shape of noise_pred[:, idx]: {noise_pred[:, idx].shape}")
            # print(f"Shape of _noise_pred: {_noise_pred.shape}")
            # print(f"Shape of weight: {weight[:, None, None, None].shape}")
            
            noise_pred[:, idx] += _noise_pred * weight[:, None, None, None]
            noise_pred_cnt[idx] += weight
        noise_pred.div_(noise_pred_cnt[:, None, None, None])
        
        return noise_pred
    
def compute_snr(noise_scheduler, timesteps):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    
    # Expand the tensors.
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)
    
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)
    
    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def log_validation(
    net,
    mimicmotion_models,
    accelerator,
    width,
    height,
    infer_config,
    clip_length=24,
    generator=None,
    ):
    logger.info("Running validation... ")
    
    ori_net = accelerator.unwrap_model(net)
    unet = ori_net.unet
    pose_net = ori_net.pose_guider
    control_net = ori_net.control_net
    
    if generator is None:
        generator = torch.manual_seed(42)
    tmp_unet = copy.deepcopy(unet)
    tmp_unet = tmp_unet.to(dtype=torch.float16)
    
    tmp_pose_net = copy.deepcopy(pose_net)
    tmp_pose_net = tmp_pose_net.to(dtype=torch.float16)
    
    tmp_control_net = copy.deepcopy(control_net)
    tmp_control_net = tmp_control_net.to(dtype=torch.float16)
    
    tmp_mimicmotion_models = copy.deepcopy(mimicmotion_models)
    tmp_mimicmotion_models = tmp_mimicmotion_models.to(dtype=torch.float16)
    
    pipeline = MimicMotionPipeline2(
        vae=tmp_mimicmotion_models.vae, 
        image_encoder=tmp_mimicmotion_models.image_encoder, 
        unet=tmp_unet, 
        scheduler=tmp_mimicmotion_models.noise_scheduler,
        feature_extractor=tmp_mimicmotion_models.feature_extractor, 
        pose_net=tmp_pose_net,
        control_net=tmp_control_net,
    )
    
    results = []
    # log_file_path = infer_config.log_file if infer_config.log_file is not None else f"\
    # {infer_config.output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.log"

    for task in infer_config.test_case:
        pose_pixels, image_pixels, image = preprocess(
            task.ref_video_path, task.ref_image_path,  
            resolution=task.resolution, sample_stride=task.sample_stride
        )
        
        hand_images, hand_masks = preprocess_hand(
            task.ref_hand_path, task.ref_hand_mask_path, 
            task.images_depth_path, task.images_mask_path,
            image, sample_stride=task.sample_stride
        )
        
        # print(f"Image Pixels - shape: {image_pixels.shape}, dtype: {image_pixels.dtype}, device: {image_pixels.device}")
        # print(f"Pose Pixels - shape: {pose_pixels.shape}, dtype: {pose_pixels.dtype}, device: {pose_pixels.device}")
        image_pixels = image_pixels.to(device=accelerator.device, dtype=torch.float16)
        pose_pixels = pose_pixels.to(device=accelerator.device, dtype=torch.float16)
        hand_images = hand_images.to(device=accelerator.device, dtype=torch.float16)
        hand_masks = hand_masks.to(device=accelerator.device, dtype=torch.float16)
        # print(f"After Conversion - Image Pixels: {image_pixels.dtype}, device: {image_pixels.device}")
        # print(f"After Conversion - Pose Pixels: {pose_pixels.dtype}, device: {pose_pixels.device}")
        
        _video_frames = run_pipeline(
            pipeline, 
            image_pixels, pose_pixels, 
            hand_images, hand_masks, 
            device, task
        )
        
        save_to_mp4(
            _video_frames, 
            f"{infer_config.output_dir}/{os.path.basename(task.ref_video_path).split('.')[0]}" \
            f"_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4",
            fps=task.fps,
        )
        
    del tmp_unet
    del tmp_pose_net
    del tmp_control_net
    del tmp_mimicmotion_models
    del pipeline
    torch.cuda.empty_cache()

def get_add_time_ids(
    fps,
    motion_bucket_id,
    noise_aug_strength,
    dtype,
    batch_size,
    unet
    ):
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

    passed_add_embed_dim = unet.config.addition_time_embed_dim * \
        len(add_time_ids)
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, \
            but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check\
            `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )
    
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids

def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"../autodl-tmp/{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    inference_config_path = "./configs/inference/inference_stage2.yaml"
    infer_config = OmegaConf.load(inference_config_path)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    
    # TODO:Model loading from ckpt
    cond_stage_model = FrozenCLIPEmbedder().to(device=device)
    mimicmotion_models = MimicMotionModel(cfg.base_model_path).to(device=device)
    
    mimicmotion_models.vae.to(dtype=weight_dtype)
    mimicmotion_models.image_encoder.to(dtype=weight_dtype)
    
    # Freeze
    mimicmotion_models.vae.requires_grad_(False)
    mimicmotion_models.image_encoder.requires_grad_(False)
    mimicmotion_models.unet.requires_grad_(False)
    for name, param in mimicmotion_models.unet.named_parameters():
        if 'temporal_transformer_block' in name:
            param.requires_grad = True
    
    mimicmotion_models.load_state_dict(torch.load(cfg.stage1_ckpt_dir, map_location="cuda"), strict=False)
    
    control_stage_config = cfg['control_stage_config']
    control_net = ControlNet(noise_latent_channels=mimicmotion_models.unet.config.block_out_channels[0])
    
    net = Net(
        mimicmotion_models.unet,
        mimicmotion_models.pose_net,
        control_net,
    ).to(device=device)
    
    cfg.solver.gradient_checkpointing = True
    if cfg.solver.gradient_checkpointing:
        mimicmotion_models.unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate
    
    # Initialize the optimizer
    cfg.solver.use_8bit_adam = True
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )
    
    # TODO:Minicmotion
    train_dataset =  MimicMotionDataset(
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.data.n_sample_frames,
        sample_rate=cfg.data.sample_rate,
        img_scale=(1.0, 1.0),
        data_meta_paths=cfg.data.meta_paths,
        hand=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=4
    )
    
    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    
    # recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    # Begin
    # num_train_epochs = 1
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(train_dataloader):
            t_data = time.time() - t_data_start
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values_vid = batch["pixel_values_vid"].to(weight_dtype)
                source = pixel_values_vid
                with torch.no_grad():
                    video_length = pixel_values_vid.shape[1]
                    pixel_values_vid = rearrange(
                        pixel_values_vid, "b f c h w -> (b f) c h w"
                    )
                    latents = mimicmotion_models.vae.encode(pixel_values_vid).latent_dist.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device,
                    )
                    
                # Prepare timesteps
                bsz = latents.shape[0]
                
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                # timesteps, num_inference_steps = retrieve_timesteps(mimicmotion_models.noise_scheduler,\ 
                # cfg.num_inference_steps, device, None)
                
                pixel_values_pose = batch["pixel_values_pose"]  # (bs, f, c, H, W)
                pixel_values_hand = batch["pixel_values_hand"]  # (bs, f, c, H, W)
                pixel_values_hand_mask = batch["pixel_values_hand_mask"]  # (bs, f, c, H, W)
                # .to(device="cuda", dtype=weight_dtype)
                
                pixel_values_pose = pixel_values_pose.transpose(
                    1, 2
                ).to(device=device)  # (bs, c, f, H, W)
                pixel_values_hand = pixel_values_hand.transpose(
                    1, 2
                ).to(device=device)  # (bs, c, f, H, W)
                pixel_values_hand_mask = pixel_values_hand_mask.transpose(
                    1, 2
                ).to(device=device)  # (bs, c, f, H, W)
                    
                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_image_list = []
                for batch_idx, (ref_img, clip_img) in enumerate(
                    zip(
                        batch["pixel_values_ref_img"],
                        batch["clip_ref_img"],
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)
                
                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(
                        dtype= mimicmotion_models.vae.dtype, device= mimicmotion_models.vae.device
                    )
                    ref_image_latents = mimicmotion_models.vae.encode(ref_img).latent_dist.mode()
                    num_frames = latents.shape[2]
                    ref_image_latents  = ref_image_latents.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
                    # ref_image_latents =  mimicmotion_models.vae.encode(
                    #     ref_img
                    # ).latent_dist.sample()  # (bs, d, 64, 64)
                    # ref_image_latents = ref_image_latents * 0.18215
                    
                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=mimicmotion_models.image_encoder.dtype, device=mimicmotion_models.image_encoder.device
                    )
                    clip_img = clip_img.to(device="cuda", dtype=weight_dtype)
                    clip_image_embeds =  mimicmotion_models.image_encoder(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    clip_image_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
                
                # TODO: add noise
                # print(f"Timesteps: {timesteps}")
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )
                
                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )
                
                noise_aug_strength = 0.02
                added_time_ids = get_add_time_ids(
                    7, # fixed
                    127, # motion_bucket_id = 127, fixed
                    noise_aug_strength, # noise_aug_strength == cond_sigmas
                    clip_image_embeds.dtype,
                    bsz,
                    mimicmotion_models.unet
                ).to(device=device, dtype=weight_dtype)
                
                # TODO: 处理负面prompt信息
                num_samples = noisy_latents.shape[0]
                c_crossattn = cond_stage_model.encode([cfg.n_prompt] * num_samples)
                
                # TODO: 处理控制图像(深度图)
                hint = pixel_values_hand
                # print(f"Shape of hint: {hint.shape}")
                # hint = hint[
                #     None,
                # ].repeat(3, axis=0)
                # hint = torch.stack([torch.tensor(hint) for _ in range(num_samples)], dim=0).to(device)
                
                # TODO: 处理控制图像(深度图)
                mask = pixel_values_hand_mask
                # print(f"mask: {mask.shape, source.shape}")
                mask = mask[None]
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1
                
                masked_image = source.permute(0, 2, 1, 3, 4) * (mask < 0.5)  # masked image is c h w
                mask = mask.squeeze(0)
                masked_image = masked_image.squeeze(0)
                # print(f"mask: {mask.shape, masked_image.shape}")
                # mask = torch.stack([torch.tensor(mask) for _ in range(num_samples)], dim=0).to("cuda")
                mask = torch.nn.functional.interpolate(mask, size=(16, 64, 64), mode='trilinear', align_corners=False)
                masked_image = torch.nn.functional.interpolate(masked_image, size=(16, 64, 64), mode='trilinear', align_corners=False)
                
                cats = torch.cat([mask, masked_image], dim=-2)

                cond = {
                    "c_concat": [cats],
                    "c_control": [hint],
                    "c_crossattn": [c_crossattn],
                }
                
                # control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_control'], 1), timesteps=t, context=cond_txt)
                # control = [c * scale for c, scale in zip(control, self.control_scales)]  # 应用控制缩放因子
                # if self.global_average_pooling:
                #     # 如果使用全局平均池化，计算控制数据的平均值
                #     control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
                # # 用扩散模型生成噪声
                # eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control,\
                #               c_concat=torch.cat(cond['c_concat'], 1), only_mid_control=self.only_mid_control)
                
                # ---- Forward!!! -----
                # print(f"Shape of noisy_latents: {noisy_latents.shape}")
                # print(f"Shape of ref_image_latents: {ref_image_latents.shape}")
                # print(f"Shape of clip_image_embeds: {clip_image_embeds.shape}")
                # print(f"Shape of pixel_values_pose: {pixel_values_pose.shape}")
                # print(f"Shape of added_time_ids: {added_time_ids.shape}")
                # print(f"Shape of timesteps: {timesteps.shape}")
                
                # print(f"Shape of cats: {cats.shape}")
                # print(f"Shape of hint: {hint.shape}")
                # print(f"Shape of c_crossattn: {timesteps.shape}")
                
                model_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    clip_image_embeds,
                    pixel_values_pose,
                    cond,
                    added_time_ids=added_time_ids,
                    uncond_fwd=uncond_fwd,
                )
                target = target.permute(0, 2, 1, 3, 4).to(device=device)
                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    # print(f"Shape of model_pred: {model_pred.shape}")
                    # print(f"Shape of target: {target.shape}")
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                if global_step % cfg.val.validation_steps == 0:
                    # if accelerator.is_main_process:
                    generator = torch.Generator(device=accelerator.device)
                    generator.manual_seed(cfg.seed)

                    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                    delete_additional_ckpt(save_dir, 1)
                    accelerator.save_state(save_path)
                    # save motion module only
                    unwrap_net = accelerator.unwrap_model(net)
                    save_checkpoint(
                        unwrap_net.unet,
                        unwrap_net.pose_guider, 
                        unwrap_net.control_net, 
                        save_dir,
                        "motion_module",
                        global_step,
                        total_limit=8,
                        # control_weight=unwrap_net.control_weight,
                    )
                    
                    # log_validation(
                    #     net=net,
                    #     mimicmotion_models=mimicmotion_models,
                    #     accelerator=accelerator,
                    #     infer_config=infer_config,
                    #     width=cfg.data.train_width,
                    #     height=cfg.data.train_height,
                    #     clip_length=cfg.data.n_sample_frames,
                    #     generator=generator,
                    # )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
                
        # save model after each epoch
        if accelerator.is_main_process and global_step % 1000 == 0 and global_step != 0:
            save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
            delete_additional_ckpt(save_dir, 1)
            accelerator.save_state(save_path)
            # save motion module only
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.unet,
                unwrap_net.pose_guider, 
                unwrap_net.control_net, 
                save_dir,
                "motion_module",
                global_step,
                total_limit=8,
                control_weight=unwrap_net.control_weight,
            )
            
            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(cfg.seed)
            
            log_validation(
                net=net,
                mimicmotion_models=mimicmotion_models,
                accelerator=accelerator,
                infer_config=infer_config,
                width=cfg.data.train_width,
                height=cfg.data.train_height,
                clip_length=cfg.data.n_sample_frames,
                generator=generator,
            )
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    
def save_checkpoint(unet, pose_net, control_net, save_dir, prefix, ckpt_num, total_limit=None, control_weight=None):
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")
    
    unet_temporal_params = {
        name: param
        for name, param in unet.named_parameters()
        if 'temporal_transformer_block' in name
    }
    pose_net_params = pose_net.state_dict()
    control_net_params = control_net.state_dict()
    
    if control_weight:
        checkpoint = {
            'unet_temporal_transformer_block': unet_temporal_params,
            'pose_net': pose_net_params,
            'control_net': control_net_params,
            'control_weight': control_weight,
        }
    else:
        checkpoint = {
            'unet_temporal_transformer_block': unet_temporal_params,
            'pose_net': pose_net_params,
            'control_net': control_net_params,
        }
    torch.save(checkpoint, save_path)
    
    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )
        
        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)
        print(f"Checkpoint saved at {save_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage2.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)