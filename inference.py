import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from PIL import Image, ImageFile
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from networks.multi_backbone import Multi_backbone_ResNet_v2

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        description="AIGC Detection Inference (NPU Version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weight", type=str, default=os.path.join(BASE_DIR, "CKPT_8_full.pth"), help="模型权重路径")
    parser.add_argument("--input_jsonl", type=str, default=os.path.join(BASE_DIR, "input_infer.jsonl"),
                        help='输入 JSONL，每行至少包含 {"img_path": "..."}')
    parser.add_argument("--output_jsonl", type=str, default=os.path.join(BASE_DIR, "output_infer.jsonl"),
                        help="输出 JSONL 路径")
    parser.add_argument("--device", type=str, default="auto", help="设备，如 auto / npu:0 / cpu")
    parser.add_argument("--gpu_id", type=str, default="0", help="device=auto 时使用的 NPU 编号")
    parser.add_argument("--arch", type=str, default="Swin_T_mix_TFS", help="模型结构名")
    parser.add_argument("--threshold", type=float, default=0.5, help="判定 ai/nature 的阈值")
    parser.add_argument("--strict_load", type=str2bool, default=True, help="是否严格校验 checkpoint 键名")
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
        print(f"[INFO] Using device: {requested}")
        return torch.device(requested)

    if requested == "cpu":
        print("[INFO] Using device: cpu")
        return torch.device("cpu")

    raise ValueError("当前脚本默认面向昇腾 NPU，device 请使用 auto / npu:0 / cpu")


def extract_state_dict(checkpoint):
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
    return checkpoint


def normalize_state_dict_keys(state_dict):
    if not state_dict:
        return state_dict
    return {
        (key.replace("module.", "", 1) if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_model(weight: str, device: torch.device, arch: str, strict: bool = True):
    print(f"[INFO] Loading model from {weight} ...")
    model = Multi_backbone_ResNet_v2(arch=arch)

    checkpoint = torch.load(weight, map_location="cpu")
    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        print(
            f"[INFO] Checkpoint loaded with strict=False | "
            f"missing={list(incompatible.missing_keys)} | unexpected={list(incompatible.unexpected_keys)}"
        )

    model = model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully.")
    return model


def preprocess_image(img_path: str):
    image = Image.open(img_path).convert("RGB")

    transform_global = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    transform_patch_128 = transforms.Compose([
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
    ])
    transform_patch_64 = transforms.Compose([
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
    ])

    x1 = transform_global(image)
    x2_1 = transform_patch_128(image)
    x2_2 = transform_patch_64(image)

    x3 = torch.fft.fft2(x1.unsqueeze(0), dim=(-2, -1))
    x3 = torch.abs(x3)

    return x1.unsqueeze(0), x2_1.unsqueeze(0), x2_2.unsqueeze(0), x3


def predict(model, device: torch.device, img_path: str):
    x1, x2_1, x2_2, x3 = preprocess_image(img_path)
    x1 = x1.to(device)
    x2_1 = x2_1.to(device)
    x2_2 = x2_2.to(device)
    x3 = x3.to(device)

    with torch.no_grad():
        z_1, z_2_1, z_2_2, z_3 = model(x1, x2_1, x2_2, x3)
        z_1 = F.softmax(z_1, dim=1)
        z_2_1 = F.softmax(z_2_1, dim=1)
        z_2_2 = F.softmax(z_2_2, dim=1)
        z_3 = F.softmax(z_3, dim=1)

        z_fuse = z_1 + (z_2_1 + z_2_2) / 2 + z_3
        z_fuse = F.softmax(z_fuse, dim=1)

        prob_real = float(z_fuse[0][0].detach().cpu())
        prob_fake = float(z_fuse[0][1].detach().cpu())
    return prob_real, prob_fake


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device, args.gpu_id)
    model = load_model(args.weight, device, args.arch, args.strict_load)

    print(f"[INFO] Reading input from {args.input_jsonl} ...")
    records = []
    with open(args.input_jsonl, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"[INFO] Total {len(records)} records to process.")

    results = []
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for index, record in enumerate(records, start=1):
        img_path = record["img_path"]
        if index == 1 or index % 50 == 0 or index == len(records):
            print(f"[INFO] Processing {index}/{len(records)}")

        prob_real, prob_fake = predict(model, device, img_path)
        category = "ai" if prob_fake > args.threshold else "nature"
        results.append({
            "id": img_path,
            "category": category,
            "prob_real": prob_real,
            "prob_fake": prob_fake,
        })

    print(f"[INFO] Writing results to {args.output_jsonl} ...")
    with open(args.output_jsonl, "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"[INFO] Done! Successfully processed {len(results)} / {len(records)}")


if __name__ == "__main__":
    main()
