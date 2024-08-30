# Mimic-Hand


## Quickstart

For the initial released version of the model checkpoint, it supports generating videos with a maximum of 72 frames at a 576x1024 resolution. If you encounter insufficient memory issues, you can appropriately reduce the number of frames.

### Environment setup

Use Python 3.10 with torch 2.4.0, and conduct experiments on an L20/48G. [torch 2.4.0](https://pan.quark.cn/s/4b0305d12e81)

```
torch-2.4.0+cu121-cp310-cp310-linux_x86_64.whl
torchaudio-2.4.0+cu121-cp310-cp310-linux_x86_64.whl
torchvision-0.19.0+cu121-cp310-cp310-linux_x86_64.whl
```
```
pip install -r requirements.txt
```

### MeshGraphormer
```
cd MimicMotions/MeshGraphormer
pip install ./manopth/.
bash scripts/download_models.sh
```
The `MeshGraphormer/models` should be organized as follows

```
models/
├── graphormer_release
│   └── graphormer_hand_state_dict.bin
├── hrnet
│   ├── cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
│   └── hrnetv2_w64_imagenet_pretrained.pth
└──
```

### Download weights
If you experience connection issues with Hugging Face, you can utilize the mirror endpoint by setting the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.
Please download weights manually as follows:
```
cd MimicMotions/
mkdir -p models/DWPose
```
1. Download DWPose pretrained model: [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
    ```
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O models/DWPose/yolox_l.onnx
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O models/DWPose/dw-ll_ucoco_384.onnx
    ```

2. The SVD model can be downloaded form [stabilityai/stable-video-diffusion-img2vid-xt-1-1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1).
    ```
    git lfs install
    git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
    mkdir -p models/SVD
    mv stable-video-diffusion-img2vid-xt-1-1 models/SVD/
    ```

3. Download the pre-trained checkpoint of MimicMotion from [Huggingface](https://huggingface.co/ixaac/MimicMotion) and download the weights after retraining on tiktok data [Weights](https://pan.quark.cn/s/db4616a14ef3).
    ```
    wget -P models/ https://huggingface.co/ixaac/MimicMotion/resolve/main/MimicMotion_1-1.pth
    ```

4. Note that if openai/clip-vit-large-patch14 does not load, download it from [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14).


Finally, all the weights should be organized in models as follows

```
models/
├── DWPose
│   ├── dw-ll_ucoco_384.onnx
│   └── yolox_l.onnx
├── SVD
│   ├── stable-video-diffusion-img2vid-xt-1-1
├── MimicMotion_1-1.pth
└── motion_module-2400.pth
```

### Test

For the test, go to `test_run.ipynb` and sequentially generate the hand depth map features of the video and reference images. Demos are in `MimicMotion/assets/test_data` and `outputs` are in outputs (For the same sample, one is the original model and the other is the improved model). 


## Citation	
```bib
@article{mimicmotion2024,
  title={MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance},
  author={Yuang Zhang and Jiaxi Gu and Li-Wen Wang and Han Wang and Junqi Cheng and Yuefeng Zhu and Fangyuan Zou},
  journal={arXiv preprint arXiv:2406.19680},
  year={2024}
}
```
