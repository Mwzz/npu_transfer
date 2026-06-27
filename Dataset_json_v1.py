"""V1 数据集: 优化 FFT 分支特征提取。

优化点 (对比原版仅做 abs):
  1. fftshift — 将 DC 分量移到频谱中心
  2. log1p   — 压缩幅值动态范围 (原 DC 可达 1e5+，高频 <1)
  3. per-image per-channel 标准化 — 零均值单位方差
  4. ImageNet normalize — 对齐 ResNet50 预训练输入分布

本文件为独立完整实现，不从原 Dataset_json.py 继承，避免混淆。
"""

import json
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from lota_patch_utils import extract_lota_patch


Record = Dict[str, object]


def load_jsonl_records(jsonl_path: str) -> List[Record]:
    records: List[Record] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append({
                "img_path": str(item["img_path"]),
                "label": int(item["label"]),
            })
    return records


def compute_fft_feature_v1(
    image_1_raw: torch.Tensor,
    normalize_imagenet: transforms.Normalize,
) -> torch.Tensor:
    """优化的 FFT 特征提取。

    Args:
        image_1_raw: ToTensor 后、未 normalize 的 RGB 图像, shape (3, H, W)
        normalize_imagenet: ImageNet Normalize transform

    Returns:
        归一化后的 FFT 幅值图, shape (3, H, W)
    """
    # FFT
    freq = torch.fft.fft2(image_1_raw, dim=(-2, -1))
    mag = torch.abs(freq)

    # 1) fftshift: DC 移到中心
    mag = torch.fft.fftshift(mag, dim=(-2, -1))

    # 2) log1p 压缩动态范围
    mag = torch.log1p(mag)

    # 3) per-image per-channel 标准化
    mean = mag.mean(dim=(-2, -1), keepdim=True)
    std = mag.std(dim=(-2, -1), keepdim=True)
    mag = (mag - mean) / (std + 1e-8)

    # 4) ImageNet normalize (对齐 ResNet50 预训练分布)
    mag = normalize_imagenet(mag)

    return mag


class FakeBenchSLFDatasetV1(Dataset):
    """V1 Dataset: 与原版 FakeBenchSLFDataset 逻辑一致，仅 FFT 分支替换为 compute_fft_feature_v1。"""

    def __init__(
        self,
        records: Sequence[Record],
        img_size: int = 256,
        patch_size: int = 32,
        bit_mode: str = "scaling",
        patch_mode: str = "max",
        is_training: bool = True,
        p_gnoise: float = 0.3,
        p_jpeg: float = 0.3,
    ) -> None:
        self.records = list(records)
        self.img_size = img_size
        self.patch_size = patch_size
        self.bit_mode = bit_mode
        self.patch_mode = patch_mode
        self.is_training = is_training
        self.p_gnoise = p_gnoise
        self.p_jpeg = p_jpeg

        # image_1 / image_3 共享的 albumentations 增强（GaussNoise + JpegCompression）
        # 对齐 train_multi_v2.py 的 transformA，仅训练时启用
        if is_training and (p_gnoise > 0 or p_jpeg > 0):
            self.transform_a = A.Compose([
                A.GaussNoise(var_limit=(10, 50), mean=0, p=p_gnoise),
                A.ImageCompression(quality_lower=35, quality_upper=90, p=p_jpeg),
            ])
        else:
            self.transform_a = None

        # image_1 / image_3 的几何增强 + ToTensor
        # 训练: Resize + HFlip + VFlip + ToTensor  (对齐 aug_fun_1)
        # 验证: Resize + ToTensor
        aug_list = [transforms.Resize((img_size, img_size))]
        if is_training:
            aug_list += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ]
        aug_list.append(transforms.ToTensor())
        self.transform_global = transforms.Compose(aug_list)

        # image_1 的 ImageNet normalize (对齐 Swin-T / ResNet50 预训练输入分布)
        self.normalize_imagenet = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # image_2 (LOTA) 的后处理: ToTensor + Normalize (对齐 LOTA loader.py)
        self.transform_lota = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def _apply_transform_a(self, image: Image.Image) -> Image.Image:
        """对 PIL 原图应用 albumentations 噪声/JPEG 增强。"""
        if self.transform_a is None:
            return image
        img_np = np.array(image)
        img_np = self.transform_a(image=img_np)["image"]
        return Image.fromarray(img_np)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]
        image = Image.open(item["img_path"]).convert("RGB")
        label = int(item["label"])

        # ---- image_2 (LOTA): 独立处理，对齐 LOTA 原生，不加增强 ----
        image_lota = extract_lota_patch(
            image=image,
            img_size=self.img_size,
            patch_size=self.patch_size,
            bit_mode=self.bit_mode,
            patch_mode=self.patch_mode,
            is_training=self.is_training,
        )
        image_2 = self.transform_lota(image_lota)

        # ---- image_1 / image_3: 共享增强流程，对齐 train_multi_v2.py ----
        # 1) [训练] GaussNoise + JpegCompression (作用在原图阶段)
        image_aug = self._apply_transform_a(image) if self.is_training else image
        # 2) Resize + [训练] HFlip + VFlip + ToTensor (此处不含 normalize)
        image_1_raw = self.transform_global(image_aug)
        # 3) image_3 = V1 优化 FFT 特征 (fftshift + log1p + 标准化 + ImageNet normalize)
        image_3 = compute_fft_feature_v1(image_1_raw, self.normalize_imagenet)
        # 4) image_1 加 ImageNet normalize (对齐 Swin-T 预训练)
        image_1 = self.normalize_imagenet(image_1_raw)

        return {
            "image_1": image_1,
            "image_2": image_2,
            "image_3": image_3,
            "label": torch.tensor(label, dtype=torch.long),
        }


def build_datasets_from_jsonl_v1(
    train_jsonl: str,
    val_jsonl: str,
    img_size: int = 256,
    patch_size: int = 32,
    bit_mode: str = "scaling",
    patch_mode: str = "max",
) -> Tuple[FakeBenchSLFDatasetV1, FakeBenchSLFDatasetV1]:
    train_records = load_jsonl_records(train_jsonl)
    val_records = load_jsonl_records(val_jsonl)

    train_dataset = FakeBenchSLFDatasetV1(
        records=train_records,
        img_size=img_size,
        patch_size=patch_size,
        bit_mode=bit_mode,
        patch_mode=patch_mode,
        is_training=True,
    )
    val_dataset = FakeBenchSLFDatasetV1(
        records=val_records,
        img_size=img_size,
        patch_size=patch_size,
        bit_mode=bit_mode,
        patch_mode=patch_mode,
        is_training=False,
    )
    return train_dataset, val_dataset
