import argparse
import json
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from bit_patch import bit_patch
from model import model as DeepLearningModel
import torch_npu  # noqa: F401

ImageFile.LOAD_TRUNCATED_IMAGES = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LOTA NPU inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_jsonl", type=str, default=os.path.join(BASE_DIR, "input_infer.jsonl"),
                        help="输入 JSONL，字段至少包含 img_path，可选 label")
    parser.add_argument("--output_jsonl", type=str,
                        default=os.path.join(BASE_DIR, "results", "predictions_temp.jsonl"),
                        help="推理结果输出 JSONL")
    parser.add_argument("--load", type=str, required=True, help="模型权重路径")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备，如 auto / npu:0 / cuda:0 / cpu")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="兼容原脚本保留，device=auto 时优先用这个编号")
    parser.add_argument("--val_batchsize", type=int, default=64, help="推理 batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker 数")
    parser.add_argument("--pin_memory", type=str2bool, default=False, help="是否开启 pin_memory")
    parser.add_argument("--img_height", type=int, default=256, help="输入尺寸")
    parser.add_argument("--isPatch", type=str2bool, default=True, help="是否启用 bit patch 预处理")
    parser.add_argument("--bit_mode", type=str, default="scaling", choices=["scaling", "thresholding"],
                        help="bit patch 模式")
    parser.add_argument("--patch_size", type=int, default=32, help="patch 大小")
    parser.add_argument("--patch_mode", type=str, default="max", choices=["max", "min", "random"],
                        help="patch 选择模式")
    parser.add_argument("--pretrain", type=str2bool, default=False,
                        help="是否加载 ImageNet 预训练 backbone，推理通常建议 False")
    parser.add_argument("--strict_load", type=str2bool, default=True,
                        help="是否严格校验 checkpoint 键名")
    parser.add_argument("--path_prefix_src", type=str, default=None,
                        help="若 JSONL 内是宿主机路径，可指定原始前缀做路径替换")
    parser.add_argument("--path_prefix_dst", type=str, default=None,
                        help="替换后的容器内路径前缀")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def set_random_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(device_arg: str, gpu_id: str) -> torch.device:
    requested = (device_arg or "auto").strip().lower()
    if requested == "auto":
        requested = f"npu:{gpu_id}"
    elif requested.isdigit():
        requested = f"npu:{requested}"

    if requested.startswith("npu"):
        torch.npu.set_device(requested)
        print(f"Using device: {requested}")
        return torch.device(requested)

    if requested == "cpu":
        print("Using device: cpu")
        return torch.device("cpu")

    raise ValueError("当前脚本默认面向昇腾 NPU，device 请使用 auto / npu:0 / cpu")


def remap_image_path(img_path: str, src_prefix: Optional[str], dst_prefix: Optional[str]) -> str:
    if not src_prefix or not dst_prefix:
        return img_path

    normalized_path = os.path.normpath(img_path)
    normalized_src = os.path.normpath(src_prefix)
    normalized_dst = os.path.normpath(dst_prefix)

    if normalized_path == normalized_src:
        return normalized_dst
    if normalized_path.startswith(normalized_src + os.sep):
        relative_path = os.path.relpath(normalized_path, normalized_src)
        return os.path.normpath(os.path.join(normalized_dst, relative_path))
    return img_path


def create_preprocessing_pipeline(options):
    if options.isPatch:
        transform_func = transforms.Lambda(
            lambda img: bit_patch(
                img,
                options.img_height,
                options.bit_mode,
                options.patch_size,
                options.patch_mode,
            )
        )
    else:
        transform_func = transforms.Resize((options.img_height, options.img_height))

    return transforms.Compose([
        transform_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class GenerativeImageInferenceSet(Dataset):
    def __init__(self, jsonl_path: str, options):
        super().__init__()
        self.options = options
        self.pipeline = create_preprocessing_pipeline(options)
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line_id, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "img_path" not in data:
                    raise ValueError(f"Line {line_id} 缺少 img_path 字段")

                original_path = data["img_path"]
                resolved_path = remap_image_path(
                    original_path,
                    options.path_prefix_src,
                    options.path_prefix_dst,
                )
                label = int(data.get("label", -1))

                self.samples.append({
                    "original_path": original_path,
                    "resolved_path": resolved_path,
                    "label": label,
                })

    @staticmethod
    def _load_rgb(img_path: str) -> Image.Image:
        try:
            with open(img_path, "rb") as file:
                img = Image.open(file)
                return img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"图像读取失败: {img_path} | {exc}") from exc

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._load_rgb(sample["resolved_path"])
        processed_image = self.pipeline(image)
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)
        return processed_image, label_tensor, sample["original_path"]

    def __len__(self) -> int:
        return len(self.samples)


def get_infer_loader(options, jsonl_path: str) -> Tuple[DataLoader, int]:
    infer_dataset = GenerativeImageInferenceSet(jsonl_path, options)

    def collate_batch(batch):
        inputs = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        paths = [item[2] for item in batch]
        return inputs, labels, paths

    infer_loader = DataLoader(
        infer_dataset,
        batch_size=options.val_batchsize,
        shuffle=False,
        num_workers=options.num_workers,
        pin_memory=options.pin_memory,
        collate_fn=collate_batch,
    )
    return infer_loader, len(infer_dataset)


def normalize_state_dict_keys(state_dict):
    if not state_dict:
        return state_dict

    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(model_instance, checkpoint_path: str, strict: bool = True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    state_dict = normalize_state_dict_keys(state_dict)
    incompatible = model_instance.load_state_dict(state_dict, strict=strict)

    if not strict:
        print(f"Checkpoint loaded with strict=False | missing={list(incompatible.missing_keys)} | unexpected={list(incompatible.unexpected_keys)}")


def label_to_category(label: int) -> str:
    if label == 1:
        return "nature"
    if label == 0:
        return "ai"
    return "unknown"


def run_model_inference(infer_loader, neural_network, output_jsonl_path: str, device: torch.device):
    neural_network.eval()
    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Inference output will be written to: {output_jsonl_path}")

    with open(output_jsonl_path, "w", encoding="utf-8") as file:
        with torch.no_grad():
            for image_batch, target_labels, image_paths in infer_loader:
                image_batch = image_batch.to(device)
                predictions = neural_network(image_batch)

                prob_real_tensor = torch.sigmoid(predictions).flatten().detach().cpu()
                prob_real_list = prob_real_tensor.numpy().tolist()
                target_labels_list = target_labels.cpu().numpy().tolist()

                for img_path, prob_real, target_label in zip(image_paths, prob_real_list, target_labels_list):
                    record = {
                        "id": img_path,
                        "category": label_to_category(int(target_label)),
                        "prob_real": float(prob_real),
                        "prob_fake": float(1.0 - prob_real),
                    }
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Inference finished.")


def main():
    options = parse_args()
    set_random_seed(options.seed)
    device = resolve_device(options.device, options.gpu_id)

    print(f"Preparing dataset from: {options.input_jsonl}")
    infer_loader, dataset_size = get_infer_loader(options, options.input_jsonl)
    print(f"Loaded dataset size: {dataset_size}")

    print("Initializing model...")
    network_instance = DeepLearningModel(pretrain=options.pretrain).to(device)

    print(f"Loading checkpoint: {options.load}")
    load_checkpoint(network_instance, options.load, strict=options.strict_load)

    run_model_inference(infer_loader, network_instance, options.output_jsonl, device)


if __name__ == "__main__":
    main()
