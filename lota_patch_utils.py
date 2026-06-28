import random

import numpy as np
from PIL import Image
from torchvision import transforms


def _low_bit_image(img_np: np.ndarray, bit_mode: str = "scaling") -> np.ndarray:
    """提取低 3 位 bit plane。

    scaling:      low3 * (255 // 7)  放大到可视范围
    thresholding: (low3 > 0) * 255    二值化
    """
    low3 = img_np & 0x07
    if bit_mode == "scaling":
        return (low3 * (255 // 7)).astype(np.uint8)
    return (low3 > 0).astype(np.uint8) * 255


def compute_patch_score(patch: np.ndarray) -> int:
    """计算 patch 的梯度分数（水平 + 垂直 + 双对角差异之和）。"""
    patch = patch.astype(np.int64)
    diff_horizontal = np.abs(patch[:, :-1, :] - patch[:, 1:, :]).sum()
    diff_vertical = np.abs(patch[:-1, :, :] - patch[1:, :, :]).sum()
    diff_diag_1 = np.abs(patch[:-1, :-1, :] - patch[1:, 1:, :]).sum()
    diff_diag_2 = np.abs(patch[1:, :-1, :] - patch[:-1, 1:, :]).sum()
    return int(diff_horizontal + diff_vertical + diff_diag_1 + diff_diag_2)


def extract_lota_patch(
    image: Image.Image,
    img_size: int = 256,
    patch_size: int = 32,
    bit_mode: str = "scaling",
    patch_mode: str = "max",
    is_training: bool = True,
) -> Image.Image:
    """LOTA patch 提取，对齐原始 LOTA (bit_patch.py) 流程。

    流程:
      1. 在原始分辨率上提取低 bit plane（不先 resize，保留噪声细节）
      2. 仅当图像最短边 < patch_size 时，才 resize 到 img_size（保证能裁出 patch）
      3. 采样候选 patch:
         - 训练 (is_training=True):  RandomCrop 随机采样，对齐原始 LOTA，兼做数据增强
         - 测试 (is_training=False): 固定网格采样，确定性，结果可复现
      4. 按 patch_mode (max/min/random) 选择最优 patch
      5. 将选中 patch resize 到 img_size × img_size
    """
    image = image.convert("RGB")
    image_np = np.array(image, dtype=np.uint8)
    low_bit_np = _low_bit_image(image_np, bit_mode=bit_mode)
    low_bit_img = Image.fromarray(low_bit_np)

    h, w = low_bit_np.shape[:2]
    min_len = min(h, w)
    # 原始 LOTA: 仅当最短边 < patch_size 时才 resize（保证 RandomCrop 能裁出 patch）
    if min_len < patch_size:
        low_bit_img = low_bit_img.resize((img_size, img_size), Image.BILINEAR)
        low_bit_np = np.array(low_bit_img)
        h, w = img_size, img_size

    num_patch = (img_size // patch_size) ** 2

    if is_training:
        # 训练: 随机裁剪，对齐原始 LOTA 的 RandomCrop，每个 epoch 看到不同 patch
        cropper = transforms.RandomCrop(patch_size)
        patches = [np.array(cropper(low_bit_img)) for _ in range(num_patch)]
    else:
        # 测试: 固定网格采样，确定性
        rows = max(1, h // patch_size)
        cols = max(1, w // patch_size)
        grid_pts = []
        for r in range(rows):
            for c in range(cols):
                grid_pts.append((r * patch_size, c * patch_size))
        # 网格点过多时均匀抽取 num_patch 个，保持候选数量与训练一致
        if len(grid_pts) > num_patch:
            indices = np.linspace(0, len(grid_pts) - 1, num_patch).astype(int)
            grid_pts = [grid_pts[i] for i in indices]
        patches = [
            low_bit_np[y:y + patch_size, x:x + patch_size, :]
            for (y, x) in grid_pts
        ]

    scores = [compute_patch_score(p) for p in patches]

    if patch_mode == "min":
        idx = int(np.argmin(scores))
    elif patch_mode == "random":
        idx = random.randint(0, len(patches) - 1)
    else:
        idx = int(np.argmax(scores))

    patch = patches[idx]
    return Image.fromarray(patch).resize((img_size, img_size), Image.BILINEAR)
