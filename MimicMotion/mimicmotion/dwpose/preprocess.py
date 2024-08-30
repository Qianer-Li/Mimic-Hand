import decord
import numpy as np

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor


# 定义函数获取视频中的姿势序列
def get_video_pose(
        video_path: str, 
        ref_image: np.ndarray, 
        sample_stride: int=1):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): video pose path  视频文件路径
        ref_image (np.ndarray): reference image  参考图像数组
        sample_stride (int, optional): Defaults to 1.  采样间隔，默认为1

    Returns:
        np.ndarray: sequence of video pose  视频中的姿势序列
    """
    # select ref-keypoint from reference pose for pose rescale
    # 从参考姿势中选择关键点，用于姿势重缩放
    ref_pose = dwprocessor(ref_image)  # 处理参考图像以获取姿势
    ref_keypoint_id = [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]  # 定义关键点ID列表
    
    # 过滤出有效的关键点
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if ref_pose['bodies']['score'].shape[0] > 0 and ref_pose['bodies']['score'][0][i] > 0.3]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    height, width, _ = ref_image.shape

    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))  # 使用decord读取视频
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))  # 调整采样步长以匹配视频帧率
    
    # 批量处理视频帧以获取姿势
    detected_poses = [dwprocessor(frm) for frm in vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()]
    
    # 堆叠所有检测到的有效姿势的身体部分
    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,
                      ref_keypoint_id]
    # compute linear-rescale params
    # 计算线性重缩放参数
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale 
    # 对姿势进行重缩放
    for detected_pose in detected_poses:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b
        im = draw_pose(detected_pose, height, width)  # 绘制处理后的姿势
        output_pose.append(np.array(im))
    return np.stack(output_pose)


# 定义函数处理单个图像中的姿势
def get_image_pose(ref_image):
    """process image pose

    Args:
        ref_image (np.ndarray): reference image pixel value  参考图像的像素值

    Returns:
        np.ndarray: pose visual image in RGB-mode  RGB模式下的姿势视觉图像
    """
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)  # 处理图像获取姿势
    pose_img = draw_pose(ref_pose, height, width)  # 绘制姿势图像
    return np.array(pose_img)
