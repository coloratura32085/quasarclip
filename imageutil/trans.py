import torch
import torch.nn.functional as F
import random
import math

from torch import Tensor
from torchvision import transforms


class CustomRandomHorizontalFlip(object):
    """随机水平翻转5通道图像的自定义转换类."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if not torch.is_tensor(img):
            raise TypeError(f'输入图像应该是一个 torch.Tensor，但得到的是 {type(img)}')

        if torch.rand(1).item() < self.p:
            img = torch.flip(img, dims=[2])  # 假设形状为 (C, H, W)
        return img


class CustomRandomVerticalFlip(object):
    """随机垂直翻转多通道图像的自定义转换类."""

    def __init__(self, p=0.5):
        """
        Args:
            p (float): 图像垂直翻转的概率，默认为0.5。
        """
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Tensor): 要翻转的图像，形状为 (C, H, W)。

        Returns:
            Tensor: 随机翻转后的图像。
        """
        if not torch.is_tensor(img):
            raise TypeError(f'输入图像应该是一个 torch.Tensor，但得到的是 {type(img)}')

        if torch.rand(1).item() < self.p:
            img = torch.flip(img, dims=[1])  # 假设形状为 (C, H, W)
            # print("Vertical flip performed.")
        # else:
        #     print("Vertical flip not performed.")
        return img


class CustomRandomRotation(object):
    """随机旋转多通道图像的自定义转换类，带有概率控制."""

    def __init__(self, degrees, p=0.5, resample='bilinear'):
        """
        Args:
            degrees (sequence or float or int): 旋转角度范围。如果是序列，表示范围 (min, max)。
                                               如果是一个数，则范围为 (-degrees, +degrees)。
            p (float): 执行旋转的概率。默认为0.5。
            resample (str): 插值方法，如 'nearest', 'bilinear'。默认为 'bilinear'。
        """
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        if resample not in ['nearest', 'bilinear']:
            raise ValueError(f"Unsupported resample mode: {resample}. Use 'nearest' or 'bilinear'.")
        self.resample = resample
        self.p = p  # 添加概率参数

    def __call__(self, img):
        """
        Args:
            img (Tensor): 要旋转的图像，形状为 (C, H, W)。

        Returns:
            Tensor: 随机旋转后的图像（根据概率决定是否旋转）。
        """
        if not torch.is_tensor(img):
            raise TypeError(f'输入图像应该是一个 torch.Tensor，但得到的是 {type(img)}')

        # 决定是否执行旋转
        if torch.rand(1).item() < self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            angle_rad = math.radians(angle)

            # 创建旋转矩阵
            theta = torch.tensor([
                [math.cos(angle_rad), -math.sin(angle_rad), 0],
                [math.sin(angle_rad), math.cos(angle_rad), 0]
            ], dtype=torch.float)
            theta = theta.unsqueeze(0)  # 扩展为 (1, 2, 3)

            # 生成仿射网格
            grid = F.affine_grid(theta, img.unsqueeze(0).size(), align_corners=False)

            # 选择插值模式
            if self.resample == 'bilinear':
                mode = 'bilinear'
            elif self.resample == 'nearest':
                mode = 'nearest'

            # 应用网格采样进行旋转
            rotated = F.grid_sample(img.unsqueeze(0), grid, mode=mode, padding_mode='zeros', align_corners=False)
            # print("Rotation performed.")

            return rotated.squeeze(0)
        else:
            # print("Rotation not performed.")
            return img


import torch


class CustomCenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        # 假设 sample 是一个 5 通道的图像，尺寸为 (C, H, W)
        c, h, w = sample.shape
        new_h, new_w = self.size, self.size

        # 计算裁剪区域的起始位置
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # 执行裁剪
        cropped_sample = sample[:, top:top + new_h, left:left + new_w]
        # print(f"Cropped sample shape: {cropped_sample.shape}")
        return cropped_sample


class CustomReshapePermute:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # print(f"Input tensor shape before reshape: {img.shape}")
        reshaped_tensor = img.reshape(self.size, self.size, 5)
        return reshaped_tensor.permute(2, 0, 1)  # 假设需要转置维度


class CustomExpStretchWithOffset:
    def __init__(self, a, b):
        self.a = a  # 控制指数拉伸的强度（α）
        self.b = b

    def __call__(self, tensor):
        # 执行指数拉伸并减去 1
        stretched_tensor = self.b * (torch.exp(self.a * tensor) - 1)

        # # 可以选择将图像数据归一化到 [0, 1] 或其他范围
        # stretched_tensor = stretched_tensor / stretched_tensor.max()
        # tensor_min = stretched_tensor.min()
        # tensor_max = stretched_tensor.max()
        # normalized_tensor = (stretched_tensor - tensor_min) / (tensor_max - tensor_min)

        return stretched_tensor  # 返回拉伸后的图像 Tensor


class CustomRandom:
    def __init__(self):
        pass

    def __call__(self, tensor):
        # 将原始 tensor 替换为 0 到 255 之间的随机整数
        random_tensor = torch.randint(0, 256, tensor.shape, dtype=torch.float32)  # 生成 [0, 255] 之间的随机整数
        # 执行指数拉伸并减去 1
        # stretched_tensor = self.b * (torch.exp(self.a * random_tensor) - 1)
        return random_tensor  # 返回拉伸后的图像 Tensor


class CustomExtinction:
    def __init__(self, u, g, r, i, z):
        self.u = u
        self.g = g
        self.r = r
        self.i = i
        self.z = z

    def __call__(self, tensor):
        corrected_u = tensor[0, :, :] * torch.pow(10.0, 0.4 * self.u)
        corrected_g = tensor[1, :, :] * torch.pow(10.0, 0.4 * self.g)
        corrected_r = tensor[2, :, :] * torch.pow(10.0, 0.4 * self.r)
        corrected_i = tensor[3, :, :] * torch.pow(10.0, 0.4 * self.i)
        corrected_z = tensor[4, :, :] * torch.pow(10.0, 0.4 * self.z)
        corrected_tensor = torch.stack([corrected_u, corrected_g, corrected_r, corrected_i, corrected_z], dim=0)
        return corrected_tensor



class PerChannelMinMaxNorm:
    """对单张图像 **每个通道** 做 `min‑max` 归一化到 `[0, 1]`。

    ‑ 输入需为 `float32`，数值范围任意 (常见 0‑65535 / 0‑1)。
    ‑ 若 `max == min`，分母使用 `1e‑6` 以避免除零。
    ‑ 直接实现 `__call__`，可无缝放入 `torchvision.transforms.Compose`。
    """

    def __call__(self, tensor: Tensor) -> Tensor:  # 期望形状 (C,H,W)
        if tensor.ndim != 3:
            raise ValueError("PerChannelMinMaxNorm 期望输入形状为 (C,H,W)")
        flat = tensor.view(tensor.size(0), -1)             # (C, N)
        c_min = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1)
        c_max = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1)
        scale = (c_max - c_min).clamp(min=1e-6)
        return (tensor - c_min) / scale


# imageutil/trans.py (添加到你的文件中)

import torch
import random


class CustomSmartCrop:
    """
    智能裁剪：确保类星体核心区域始终在裁剪范围内，但位置随机

    工作原理：
    1. 计算图像中心（类星体核心位置）
    2. 计算允许的随机偏移范围（确保核心在裁剪内）
    3. 随机选择偏移量
    4. 执行裁剪
    """

    def __init__(self, crop_size=32, core_size=10):
        """
        Args:
            crop_size: 裁剪后的图像大小 (default: 32)
            core_size: 类星体核心区域大小 (default: 10)
                      这个区域必须完全在裁剪范围内
        """
        self.crop_size = crop_size
        self.core_size = core_size

    def __call__(self, image):
        """
        Args:
            image: torch.Tensor of shape (C, H, W)
        Returns:
            cropped image: torch.Tensor of shape (C, crop_size, crop_size)
        """
        C, H, W = image.shape

        # 图像中心即类星体核心位置
        center_h, center_w = H // 2, W // 2

        # 核心区域的半径
        core_half = self.core_size // 2

        # 裁剪窗口的半径
        crop_half = self.crop_size // 2

        # 计算最大允许偏移量
        # 偏移后裁剪窗口仍要包含整个核心区域
        max_offset_h = crop_half - core_half
        max_offset_w = crop_half - core_half

        # 随机生成偏移量 (可以是负数，表示向左上偏移)
        offset_h = random.randint(-max_offset_h, max_offset_h)
        offset_w = random.randint(-max_offset_w, max_offset_w)

        # 计算裁剪中心（在原中心基础上偏移）
        crop_center_h = center_h + offset_h
        crop_center_w = center_w + offset_w

        # 计算裁剪边界
        top = crop_center_h - crop_half
        bottom = crop_center_h + crop_half
        left = crop_center_w - crop_half
        right = crop_center_w + crop_half

        # 边界检查和调整
        # 如果超出图像边界，将裁剪窗口整体移动到边界内
        if top < 0:
            bottom = bottom - top
            top = 0
        if left < 0:
            right = right - left
            left = 0
        if bottom > H:
            top = top - (bottom - H)
            bottom = H
        if right > W:
            left = left - (right - W)
            right = W

        # 确保裁剪大小正确
        if bottom - top != self.crop_size:
            bottom = top + self.crop_size
        if right - left != self.crop_size:
            right = left + self.crop_size

        # 执行裁剪
        cropped = image[:, top:bottom, left:right]

        # 安全检查：如果尺寸不对（极端边界情况），进行填充
        if cropped.shape[1] != self.crop_size or cropped.shape[2] != self.crop_size:
            padded = torch.zeros(C, self.crop_size, self.crop_size, dtype=image.dtype, device=image.device)
            h_actual, w_actual = cropped.shape[1], cropped.shape[2]
            padded[:, :h_actual, :w_actual] = cropped
            cropped = padded

        return cropped


class CustomRandomMask:
    """
    为掩码重构任务创建随机掩码

    掩码策略：
    - 随机遮盖一定比例的像素
    - 被遮盖的像素值会在训练时被设为 0
    - 模型需要根据未遮盖的像素重构被遮盖的部分
    """

    def __init__(self, mask_ratio=0.75):
        """
        Args:
            mask_ratio: 遮盖比例 (default: 0.75，即 75% 的像素被遮盖)
        """
        self.mask_ratio = mask_ratio

    def __call__(self, image):
        """
        Args:
            image: torch.Tensor of shape (C, H, W)
        Returns:
            mask: torch.Tensor of shape (C, H, W)，1 表示遮盖，0 表示保留
        """
        C, H, W = image.shape

        # 生成随机掩码 (每个像素独立随机)
        mask = torch.rand(1, H, W, device=image.device) < self.mask_ratio

        # 扩展到所有通道（5个通道使用相同的掩码）
        mask = mask.expand(C, -1, -1).float()

        return mask
