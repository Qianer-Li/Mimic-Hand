import logging

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..modules.unet import UNetSpatioTemporalConditionModel
from ..modules.pose_net import PoseNet
from ..modules.control_net import ControlNet
from ..pipelines.pipeline_mimicmotion import MimicMotionPipeline
from ..pipelines.pipeline_mimicmotion2 import MimicMotionPipeline2

logger = logging.getLogger(__name__)

class MimicMotionModel(torch.nn.Module):
    def __init__(self, base_model_path):
        """construnct base model components and load pretrained svd model except pose-net
            初始化模型组件并加载预训练的SVD模型，除了pose-net
        Args:
            base_model_path (str): pretrained svd model path
        """
        super().__init__()
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet"))
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae").half()  # 加载并将模型转为半精度以节省内存
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder")  # 加载CLIP视觉模型
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")    # 加载噪声调度器
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor") # 加载CLIP特征提取器
        self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0])
        
def create_pipeline(infer_config, device):
    """create mimicmotion pipeline and load pretrained weight

    Args:
        infer_config (str): 
        device (str or torch.device): "cpu" or "cuda:{device_id}"
    """
    # 创建模型实例，转移到指定设备上，并设置为评估模式
    print(infer_config.base_model_path)
    mimicmotion_models = MimicMotionModel(infer_config.base_model_path).to(device=device).eval()
    
    # 加载预训练权重，非严格模式可能允许某些权重未加载
    mimicmotion_models.load_state_dict(torch.load(infer_config.ckpt_path, map_location=device), strict=False)
    
    # 创建MimicMotion管道，将所有模型组件组装起来
    pipeline = MimicMotionPipeline(
        vae=mimicmotion_models.vae, 
        image_encoder=mimicmotion_models.image_encoder, 
        unet=mimicmotion_models.unet, 
        scheduler=mimicmotion_models.noise_scheduler,
        feature_extractor=mimicmotion_models.feature_extractor, 
        pose_net=mimicmotion_models.pose_net
    )
    return pipeline

def add_prefix(param_name, prefix_to_add):
    return f"{prefix_to_add}.{param_name}"

class MimicMotionModel2(torch.nn.Module):
    def __init__(self, base_model_path, control_stage_config=None, control_weight=None):
        super().__init__()
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet"))
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae").half()  # 加载并将模型转为半精度以节省内存
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder")  # 加载CLIP视觉模型
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")    # 加载噪声调度器
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor") # 加载CLIP特征提取器
        self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0])
        self.control_net = ControlNet(noise_latent_channels=self.unet.config.block_out_channels[0])
        # self.control_weight = torch.nn.Parameter(torch.tensor(0.1)) 
        
def create_pipeline2(infer_config, device):
    """create mimicmotion pipeline and load pretrained weight

    Args:
        infer_config (str): 
        device (str or torch.device): "cpu" or "cuda:{device_id}"
    """
    # 创建模型实例，转移到指定设备上，并设置为评估模式
    print(infer_config.base_model_path)
    # control_stage_config = infer_config['control_stage_config']
    mimicmotion_models = MimicMotionModel2(infer_config.base_model_path).to(device=device).eval()
    
    # 加载预训练权重
    mimicmotion_models.load_state_dict(torch.load(infer_config.ckpt_path, map_location=device), strict=False)
    
    # 加载mimicmotion预训练权重
    # ckpt_path = infer_config.ckpt_path
    # checkpoint = torch.load(ckpt_path, map_location=device)
        
    model_state_dict = mimicmotion_models.state_dict()
    
    stage2_checkpoint = torch.load(infer_config.stage2_path, map_location=device)
    temporal_transformer_params = {
        k: v for k, v in stage2_checkpoint['unet_temporal_transformer_block'].items()
    }
    pose_net_params = {
    k: v for k, v in stage2_checkpoint['pose_net'].items()
    }
    control_net_params = {
        k: v for k, v in stage2_checkpoint['control_net'].items()
    }
    
    # for param_name, param_value in model_state_dict.items():
    #     print(f"Parameter Name: {param_name}")
    #     print(f"Parameter Shape: {param_value.shape}")
    #     print(f"Parameter Value (first few elements): {param_value.flatten()[:10]}")  # 打印前10个元素
    #     print("-" * 40)  # 分隔线
    # for param_name in model_state_dict.keys():
    #     print(param_name)
    print(model_state_dict.keys())
    for param_name, param_value in temporal_transformer_params.items():
        modelf_param_name = add_prefix(param_name, 'unet')
        # print('modelf_param_name：', modelf_param_name)
        if modelf_param_name in model_state_dict.keys():
            if model_state_dict[modelf_param_name].shape == param_value.shape:
                model_state_dict[modelf_param_name].copy_(param_value)
            else:
                print(f"Shape mismatch for parameter {param_name}: "
                      f"model shape {model_state_dict[modelf_param_name].shape}, "
                      f"checkpoint shape {param_value.shape}")
        else:
            print(f"Parameter {param_name} not found in model's state_dict")
    
    #  pose_net 部分的参数
    for param_name, param_value in pose_net_params.items():
        modelf_param_name = add_prefix(param_name, 'pose_net')
        # print('modelf_param_name：', modelf_param_name)
        if modelf_param_name in model_state_dict.keys():
            if model_state_dict[modelf_param_name].shape == param_value.shape:
                model_state_dict[modelf_param_name].copy_(param_value)
            else:
                print(f"Shape mismatch for parameter {param_name}: "
                      f"model shape {model_state_dict[modelf_param_name].shape}, "
                      f"checkpoint shape {param_value.shape}")
        else:
            print(f"Parameter {param_name} not found in model's state_dict")
    
    #  control_net 部分的参数
    for param_name, param_value in control_net_params.items():
        # modelf_param_name = add_prefix(param_name, 'control_net')
        if modelf_param_name in model_state_dict.keys():
            if model_state_dict[modelf_param_name].shape == param_value.shape:
                model_state_dict[modelf_param_name].copy_(param_value)
            else:
                print(f"Shape mismatch for parameter {param_name}: "
                      f"model shape {model_state_dict[modelf_param_name].shape}, "
                      f"checkpoint shape {param_value.shape}")
        else:
            print(f"Parameter {param_name} not found in model's state_dict")
    
    
    # 将更新后的 state_dict 加载到模型中
    mimicmotion_models.load_state_dict(model_state_dict, strict=False)
    
    # 创建MimicMotion管道，将所有模型组件组装起来
    pipeline = MimicMotionPipeline2(
        vae=mimicmotion_models.vae, 
        image_encoder=mimicmotion_models.image_encoder, 
        unet=mimicmotion_models.unet, 
        scheduler=mimicmotion_models.noise_scheduler,
        feature_extractor=mimicmotion_models.feature_extractor, 
        pose_net=mimicmotion_models.pose_net,
        control_net=mimicmotion_models.control_net,
    )
    return pipeline