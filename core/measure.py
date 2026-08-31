# -*- coding: utf-8 -*-
"""颗粒量测：形态指标计算、颗粒着色与结果叠加。"""
import cv2
import numpy as np


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
    pixel_size: mm/pixel

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
    """查询标签图 (x, y) 处的颗粒 id（背景返回 None）。"""
    h, w = markers.shape
    if 0 <= x < w and 0 <= y < h:
        val = int(markers[y, x])
        if val > 1:
            return val
    return None


def colorize_particles(markers):
    """将实例标签图着色为随机彩色图（BGR）。"""
    h, w = markers.shape
    output = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(1)
    for i in np.unique(markers):
        if i <= 1:
            continue
        color = rng.integers(0, 255, 3)
        output[markers == i] = color

    return output


def overlay_markers(image, markers, alpha=0.5, random_color=True, save_path=None):
    """将 markers 标签图叠加显示在原图上，返回叠加后的图片。

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
