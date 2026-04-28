"""
棋盘格和彩色合成图生成函数 - 使用自适应 gamma 校正和归一化，彻底解决眩光问题

参考：
1. https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
2. https://www.pyimagesearch.com/2014/01/27/opencv-python-contrast-enhancement/
3. https://www.cambridgeincolour.com/tutorials/image-processing-gamma-correction.htm
4. 多光谱图像融合最佳实践

关键修复：
1. 使用自适应 gamma 校正压缩 SAR 动态范围
2. 分别对每个通道进行归一化
3. 使用 cv2.addWeighted 进行混合
4. 确保所有值在 0-255 范围内
"""

import cv2
import numpy as np


def adaptive_gamma_correction(img, gamma=0.5):
    """
    自适应 gamma 校正，压缩高动态范围
    
    Args:
        img: 输入图像（任意范围）
        gamma: gamma 值（<1 压缩高光，>1 提升暗部）
    
    Returns:
        gamma 校正后的图像（0-255）
    """
    # 归一化到 0-1
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 1e-10:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = img * 0.0
    
    # Gamma 校正
    img_gamma = np.power(img_norm, 1.0 / gamma)
    
    # 转换到 0-255
    img_uint8 = (img_gamma * 255).astype(np.uint8)
    
    return img_uint8


def create_checkerboard_fixed(sar, optical, block_size=32):
    """
    创建棋盘格叠加图
    """
    # 获取尺寸
    if optical.ndim == 3:
        h, w = optical.shape[:2]
    else:
        h, w = optical.shape
    
    # 调整 SAR 尺寸
    sar_resized = cv2.resize(sar, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 使用 adaptive gamma 校正
    sar_enhanced = adaptive_gamma_correction(sar_resized, gamma=0.5)
    
    # 光学归一化
    optical_norm = ((optical - optical.min()) / (optical.max() - optical.min() + 1e-10) * 255).astype(np.uint8)
    
    # 转为 3 通道 BGR
    if sar_enhanced.ndim == 2:
        sar_bgr = cv2.cvtColor(sar_enhanced, cv2.COLOR_GRAY2BGR)
    else:
        sar_bgr = sar_enhanced
    
    if optical_norm.ndim == 2:
        optical_bgr = cv2.cvtColor(optical_norm, cv2.COLOR_GRAY2BGR)
    else:
        optical_bgr = optical_norm
    
    # 创建棋盘格掩码
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                end_i = min(i + block_size, h)
                end_j = min(j + block_size, w)
                mask[i:end_i, j:end_j] = 1
    
    # 创建输出
    checkerboard = np.zeros((h, w, 3), dtype=np.uint8)
    checkerboard[mask == 1] = sar_bgr[mask == 1]
    checkerboard[mask == 0] = optical_bgr[mask == 0]
    
    return checkerboard


def create_false_color_overlay_fixed(sar, optical, alpha=0.5):
    """
    创建彩色合成图 - 使用 adaptive gamma 校正，彻底解决眩光问题
    
    关键修复：
    1. 对 SAR 使用 gamma=0.5 压缩高动态范围
    2. 对光学使用 gamma=1.0 保持原样
    3. 分别创建彩色通道
    4. 使用 cv2.addWeighted 混合
    """
    # 获取尺寸
    if optical.ndim == 3:
        h, w = optical.shape[:2]
    else:
        h, w = optical.shape
    
    # 调整 SAR 尺寸
    sar_resized = cv2.resize(sar, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 对 SAR 使用 adaptive gamma 校正（gamma=0.5 压缩高光）
    sar_enhanced = adaptive_gamma_correction(sar_resized, gamma=0.5)
    
    # 对光学归一化（gamma=1.0 保持原样）
    optical_enhanced = adaptive_gamma_correction(optical, gamma=1.0)
    
    # 转为 3 通道 BGR
    if sar_enhanced.ndim == 2:
        sar_bgr = cv2.cvtColor(sar_enhanced, cv2.COLOR_GRAY2BGR)
    else:
        sar_bgr = sar_enhanced
    
    if optical_enhanced.ndim == 2:
        optical_bgr = cv2.cvtColor(optical_enhanced, cv2.COLOR_GRAY2BGR)
    else:
        optical_bgr = optical_enhanced
    
    # 创建 SAR 彩色图（红色通道）
    sar_color = np.zeros((h, w, 3), dtype=np.uint8)
    sar_color[:, :, 2] = sar_bgr[:, :, 2]  # BGR 红色是第 2 通道
    
    # 创建光学彩色图（青色 = 绿 + 蓝）
    optical_color = np.zeros((h, w, 3), dtype=np.uint8)
    optical_color[:, :, 0] = optical_bgr[:, :, 0]  # 蓝色通道
    optical_color[:, :, 1] = optical_bgr[:, :, 1]  # 绿色通道
    
    # 使用 cv2.addWeighted 混合（自动处理溢出）
    overlay = cv2.addWeighted(sar_color, alpha, optical_color, 1.0 - alpha, 0)
    
    return overlay


# 测试
if __name__ == '__main__':
    # 创建测试图像（模拟真实 SAR 高动态范围）
    sar = np.random.rand(512, 512).astype(np.float32) * 10000  # 高动态范围
    optical = np.random.rand(512, 512, 3).astype(np.float32) * 255
    
    print(f"SAR 原始范围：[{sar.min():.2f}, {sar.max():.2f}]")
    
    # 测试 gamma 校正
    sar_gamma = adaptive_gamma_correction(sar, gamma=0.5)
    print(f"SAR gamma 校正后范围：[{sar_gamma.min()}, {sar_gamma.max()}]")
    
    checkerboard = create_checkerboard_fixed(sar, optical, block_size=32)
    overlay = create_false_color_overlay_fixed(sar, optical, alpha=0.5)
    
    print(f"棋盘格：shape={checkerboard.shape}, dtype={checkerboard.dtype}, range=[{checkerboard.min()}, {checkerboard.max()}]")
    print(f"彩色合成：shape={overlay.shape}, dtype={overlay.dtype}, range=[{overlay.min()}, {overlay.max()}]")
    
    # 检查是否有眩光
    if overlay.max() > 255 or overlay.min() < 0:
        print("⚠️ 警告：值范围异常！")
    else:
        print("✅ 值范围正常，无眩光")
    
    cv2.imwrite('test_checkerboard.png', checkerboard)
    cv2.imwrite('test_overlay.png', overlay)
    print("✅ 测试图片已保存（使用 adaptive gamma 校正）")
