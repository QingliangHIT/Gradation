import os
import sys
import cv2
import numpy as np
import torch
from unet_project.unet_model import UNet, ResUNet
from samInstance_project.segment_anything import SamAutomaticMaskGenerator
from samInstance_project.segment_anything import sam_model_registry

# SAM 模型缓存
_sam_model = None
_sam_device = None
_SAM_CHECKPOINT = r"A:\05-Codes\Gradation\DATA\sam_vit_h_4b8939.pth"
_SAM_MODEL_TYPE = "vit_h"


_model = None
_device = None
_MODEL_PATH = r"A:\05-Codes\Gradation\DATA\resunet_best.pth"
_MODEL_NAME = "resunet"
_IMG_SIZE = 256
_MODELS = {"unet": UNet, "resunet": ResUNet}

# 计算设备偏好: "auto" / "cpu" / "cuda"
_DEVICE_PREF = "auto"

# YOLO 分割模型权重（可选，留空时 YOLO 模型不可用）
_YOLO_WEIGHTS = r"A:\05-Codes\Gradation\DATA\yolov8n-seg.pt"


def _resolve_device(pref):
    """根据偏好解析实际计算设备。"""
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_device():
    """按当前设备偏好解析实际计算设备（供模型注册表等外部模块使用）。"""
    return _resolve_device(_DEVICE_PREF)


def configure(device=None, unet_model_path=None, sam_checkpoint=None, yolo_weights=None):
    """运行时配置计算设备与模型权重路径，变更时自动失效缓存。
    device: 'auto' / 'cpu' / 'cuda'
    """
    global _model, _device, _sam_model, _sam_device
    global _MODEL_PATH, _SAM_CHECKPOINT, _DEVICE_PREF, _YOLO_WEIGHTS
    if device in ("auto", "cpu", "cuda") and device != _DEVICE_PREF:
        _DEVICE_PREF = device
        _model = None
        _sam_model = None
    if unet_model_path and os.path.isfile(unet_model_path) and unet_model_path != _MODEL_PATH:
        _MODEL_PATH = unet_model_path
        _model = None
    if sam_checkpoint and os.path.isfile(sam_checkpoint) and sam_checkpoint != _SAM_CHECKPOINT:
        _SAM_CHECKPOINT = sam_checkpoint
        _sam_model = None
    if yolo_weights is not None and yolo_weights != _YOLO_WEIGHTS:
        _YOLO_WEIGHTS = yolo_weights
    return get_config()


def get_config():
    """返回当前推理配置。"""
    return {
        "device": _DEVICE_PREF,
        "unet_model_path": _MODEL_PATH,
        "sam_checkpoint": _SAM_CHECKPOINT,
        "yolo_weights": _YOLO_WEIGHTS,
    }


def _load_model(model_path=None, model_name=None):
    global _model, _device
    if _model is not None:
        return
    _device = _resolve_device(_DEVICE_PREF)
    name = model_name or _MODEL_NAME
    _model = _MODELS[name](in_channels=3, num_classes=2).to(_device)
    path = model_path or _MODEL_PATH
    checkpoint = torch.load(path, map_location=_device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        _model.load_state_dict(checkpoint["model_state_dict"])
    else:
        _model.load_state_dict(checkpoint)
    _model.eval()


def unet_predict(img, model_path=None, model_name=None):
    """
    输入: BGR 彩色图 (H, W, 3) uint8 或 灰度图 (H, W) uint8
    输出: 二值图 (H, W) uint8, 颗粒=255
    """
    _load_model(model_path, model_name)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    resized = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE))
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(_device)

    with torch.no_grad():
        out = _model(tensor)
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()

    pred = (pred * 255).astype(np.uint8)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred


def segment_particles(binary, dist_thresh_ratio=0.45, kernel_size=3,
                      close_iterations=0, open_iterations=2):
    img_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(dist, dist_thresh_ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_color, markers)

    return markers


def segment_particles_unet(binary, dist_thresh_ratio=0.4, kernel_size=3,
                            close_iterations=2, open_iterations=2):
    """
    UNet 分割流程:
    1. UNet 预测二值掩膜
    2. 形态学清理
    3. Watershed 实例分割

    输入: BGR 彩色图 (H, W, 3) uint8 或 灰度图 (H, W) uint8
    返回: markers 标签图
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    img_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    sure_bg = cv2.dilate(binary, kernel, iterations=3)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, dist_thresh_ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    # markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_color, markers)

    return markers


def _shape_features(cnt, area, pixel_size):
    """计算单个颗粒轮廓的二维形态指标（12项体系，简化实现）。
    含: 周长、最大Feret径、Feret长宽比、偏心率、圆形度、圆整度、
    实心度、凸性、旋转矩形填充率、径向距离变异系数、棱角性指数、显著角点密度。
    """
    feats = {}
    perimeter = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perim = cv2.arcLength(hull, True)
    (rw, rh) = cv2.minAreaRect(cnt)[1]
    rect_area = rw * rh

    feats["perimeter_mm"] = perimeter * pixel_size

    # 最大 Feret 径（凸包顶点两两距离近似）
    hpts = hull.reshape(-1, 2).astype(np.float32)
    if len(hpts) > 120:
        hpts = hpts[np.linspace(0, len(hpts) - 1, 120).astype(int)]
    if len(hpts) >= 2:
        d = hpts[:, None, :] - hpts[None, :, :]
        feret_max = float(np.sqrt((d ** 2).sum(-1)).max())
    else:
        feret_max = 0.0
    feats["feret_major_mm"] = feret_max * pixel_size
    short = max(min(rw, rh), 1e-6)
    feats["feret_ratio"] = round(feret_max / short, 3) if feret_max > 0 else 1.0

    # 偏心率（等效椭圆）
    ecc = 0.0
    if len(cnt) >= 5:
        try:
            (ma, MA) = sorted(cv2.fitEllipse(cnt)[1])
            if MA > 0:
                ecc = float(np.sqrt(max(0.0, 1.0 - (ma / MA) ** 2)))
        except cv2.error:
            pass
    feats["eccentricity"] = round(ecc, 3)

    feats["circularity"] = round(min(4 * np.pi * area / perimeter ** 2, 1.0), 3) if perimeter > 0 else 0.0
    feats["roundness"] = round(4 * area / (np.pi * feret_max ** 2), 3) if feret_max > 0 else 0.0
    feats["solidity"] = round(area / hull_area, 3) if hull_area > 0 else 0.0
    feats["convexity"] = round(hull_perim / perimeter, 3) if perimeter > 0 else 0.0
    feats["rect_fill"] = round(area / rect_area, 3) if rect_area > 0 else 0.0

    # 径向距离变异系数（质心至轮廓各点距离的离散程度）
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        pts = cnt.reshape(-1, 2).astype(np.float32)
        r = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        feats["radial_cv"] = round(float(r.std() / r.mean()), 3) if r.mean() > 0 else 0.0
    else:
        feats["radial_cv"] = 0.0

    # 棱角性指数与显著角点密度（轮廓转角统计，简化实现）
    feats["angularity"] = 0.0
    feats["corner_density"] = 0.0
    pts = cnt.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    if n >= 6 and perimeter > 0:
        step = max(1, n // 160)
        s = pts[::step]
        if len(s) >= 3:
            if np.array_equal(s[0], s[-1]):
                s = s[:-1]
            s = np.vstack([s, s[:1], s[:2]])
            v = np.diff(s, axis=0)
            ang = np.arctan2(v[:, 1], v[:, 0])
            dtheta = np.abs(np.angle(np.exp(1j * np.diff(ang))))
            dtheta_deg = np.degrees(dtheta)
            sig = dtheta_deg[dtheta_deg > 15]
            feats["angularity"] = round(float(sig.sum() / 360.0), 3)
            feats["corner_density"] = round(len(sig) / perimeter, 4)
    return feats


def measure_particles(markers, pixel_size=0.05, min_area=20):
    """
    pixel_size:
        mm/pixel

    返回:
        每个颗粒的尺寸与二维形态参数列表（含 12 项形态指标）
    """

    particles = []
    ids = np.unique(markers)
    for i in ids:
        if i <= 1:
            continue
        mask = np.uint8(markers == i)
        area = cv2.countNonZero(mask)
        if area < min_area:
            continue
        cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnt) == 0:
            continue
        cnt = cnt[0]
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        diameter = np.sqrt(4 * area / np.pi)
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        particle = {
            "id": i,
            "area_pixel": area,
            "area_mm2": area * (pixel_size ** 2),
            "diameter_mm": diameter * pixel_size,
            "length_mm": max(w, h) * pixel_size,
            "width_mm": min(w, h) * pixel_size,
            "cx": cx,
            "cy": cy
        }
        particle.update(_shape_features(cnt, area, pixel_size))
        particles.append(particle)

    return particles


def find_particle_at(markers, x, y):
    h, w = markers.shape
    if 0 <= x < w and 0 <= y < h:
        val = int(markers[y, x])
        if val > 1:
            return val
    return None


def colorize_particles(markers):
    h, w = markers.shape
    output = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(1)
    for i in np.unique(markers):
        if i <= 1:
            continue
        color = rng.integers(0, 255, 3)
        output[markers == i] = color

    return output


def _load_sam_model():
    """加载 SAM 模型"""
    global _sam_model, _sam_device
    if _sam_model is not None:
        return _sam_model

    # 添加 SAM 项目路径（使用当前项目内的 segment_anything）
    sam_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "samInstance_project",
    )
    if sam_path not in sys.path:
        sys.path.insert(0, sam_path)

    _sam_device = _resolve_device(_DEVICE_PREF)
    _sam_model = sam_model_registry[_SAM_MODEL_TYPE](checkpoint=_SAM_CHECKPOINT)
    _sam_model.to(device=_sam_device)
    _sam_model.eval()
    return _sam_model


def sam_segment(
    img,
    points_per_side=4,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
):
    """
    使用 SAM 进行实例分割

    输入: BGR 彩色图 (H, W, 3) uint8
    返回: markers 标签图 (H, W) int32，每个实例有唯一标签
    """
    # 加载模型
    sam = _load_sam_model()

    # BGR -> RGB
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建 mask 生成器
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )

    # 生成 masks
    masks = mask_generator.generate(img_rgb)

    # 将 masks 转换为 markers 标签图
    h, w = img_rgb.shape[:2]
    markers = np.zeros((h, w), dtype=np.int32)

    # 按面积从大到小排序，大物体优先分配标签
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # 实例标签从 2 开始（与分水岭一致，1 为背景约定），避免首个颗粒被测量环节跳过
    for idx, mask_dict in enumerate(sorted_masks, start=2):
        seg = mask_dict['segmentation']
        if seg.dtype != bool:
            seg = seg.astype(bool)
        # 只标记尚未被分配的区域
        markers[seg & (markers == 0)] = idx

    # SAM 不产生独立的二值掩膜（返回 None，界面不展示二值掩膜图层）
    return None, markers


def overlay_masks(image, masks, alpha=0.5, random_color=True, save_path=None):
    """
    将掩码叠加显示在原图上，返回叠加后的图片。

    Args:
        image: 原图 (H, W, 3) uint8 RGB格式
        masks: mask列表，每个元素含 'segmentation' 键
        alpha: 掩码透明度，0=全透明 1=不透明
        random_color: 是否用随机颜色
        save_path: 若指定则保存图片到该路径

    Returns:
        overlay: 叠加后的图片 (H, W, 3) uint8
    """
    overlay = image.copy()
    if len(masks) == 0:
        return overlay

    sorted_anns = sorted(masks, key=lambda x: x['area'], reverse=True)
    for ann in sorted_anns:
        m = ann['segmentation']
        if m.dtype != bool:
            m = m.astype(bool)
        if random_color:
            color = (np.random.random(3) * 255).astype(np.uint8)
        else:
            color = np.array([128, 128, 128], dtype=np.uint8)
        overlay[m] = (overlay[m] * (1 - alpha) + color * alpha).astype(np.uint8)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"已保存: {save_path}")

    return overlay


def overlay_markers(image, markers, alpha=0.5, random_color=True, save_path=None):
    """
    将 markers 标签图叠加显示在原图上，返回叠加后的图片。

    Args:
        image: 原图 (H, W, 3) uint8 BGR格式
        markers: 标签图 (H, W) int32，每个实例有唯一标签
        alpha: 掩码透明度，0=全透明 1=不透明
        random_color: 是否用随机颜色
        save_path: 若指定则保存图片到该路径

    Returns:
        overlay: 叠加后的图片 (H, W, 3) uint8 RGB格式
    """
    # BGR -> RGB
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()

    if markers is None:
        return overlay

    # 为每个实例分配随机颜色
    rng = np.random.default_rng(42)
    unique_ids = np.unique(markers)
    
    for instance_id in unique_ids:
        if instance_id <= 0:  # 跳过背景
            continue
        
        mask = (markers == instance_id)
        if random_color:
            color = (rng.random(3) * 255).astype(np.uint8)
        else:
            color = np.array([128, 128, 128], dtype=np.uint8)
        
        overlay[mask] = (overlay[mask] * (1 - alpha) + color * alpha).astype(np.uint8)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"已保存: {save_path}")

    return overlay