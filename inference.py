import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLF_SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "SLF")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SLF_SRC_DIR not in sys.path:
    sys.path.insert(0, SLF_SRC_DIR)

from SLF_model import SLFModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="SLF NPU Inference Script",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto / npu:0 / cpu")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--bit_mode", type=str, default="scaling", choices=["scaling", "thresholding"])
    parser.add_argument("--patch_mode", type=str, default="max", choices=["max", "min", "random"])
    parser.add_argument("--data_version", type=int, default=0, choices=[0, 1])
    return parser


def resolve_device(device_arg: str, gpu_id: int) -> torch.device:
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

    raise ValueError("device 仅支持 auto / npu:0 / cpu")


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        normalized[new_k] = v
    return normalized


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
        if "model" in checkpoint_obj:
            return checkpoint_obj["model"]
    return checkpoint_obj


def load_model(ckpt_path: str, device: torch.device):
    model = SLFModel(pretrained_backbone=False)
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def predict(model, loader, device):
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            x1 = batch["image_1"].to(device, non_blocking=True)
            x2 = batch["image_2"].to(device, non_blocking=True)
            x3 = batch["image_3"].to(device, non_blocking=True)
            img_paths = batch["img_path"]

            z1, z2, z3 = model(x1, x2, x3)
            p1 = F.softmax(z1, dim=1)
            p2 = F.softmax(z2, dim=1)
            p3 = F.softmax(z3, dim=1)
            probs = (p1 + p2 + p3)
            probs = probs / probs.sum(dim=1, keepdim=True)
            preds = torch.argmax(probs, dim=1)

            preds_np = preds.cpu().numpy().tolist()
            probs_np = probs.cpu().numpy().tolist()

            for img_path, pred_label, prob_vec in zip(img_paths, preds_np, probs_np):
                results.append(
                    {
                        "img_id": img_path,
                        "is_generated": int(pred_label),
                        "prob_real": float(prob_vec[0]),
                        "prob_fake": float(prob_vec[1]),
                    }
                )
    return results


def save_results(results, output_path: str):
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved inference results to: {output_path}")


def main():
    args = build_parser().parse_args()

    if args.data_version == 1:
        from Dataset_json_v1 import FakeBenchSLFDatasetV1, load_jsonl_records

        BaseDataset = FakeBenchSLFDatasetV1
    else:
        from Dataset_json import FakeBenchSLFDataset, load_jsonl_records

        BaseDataset = FakeBenchSLFDataset

    class InferDataset(BaseDataset):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            item["img_path"] = self.records[idx]["img_path"]
            return item

    device = resolve_device(args.device, args.gpu_id)
    print(f"Dataset version: {args.data_version}")

    records = load_jsonl_records(args.input_jsonl)
    dataset = InferDataset(
        records=records,
        img_size=args.img_size,
        patch_size=args.patch_size,
        bit_mode=args.bit_mode,
        patch_mode=args.patch_mode,
        is_training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    print(f"Loaded dataset: {len(dataset)} samples from {args.input_jsonl}")

    model = load_model(args.ckpt, device)
    print(f"Loaded checkpoint from: {args.ckpt}")

    results = predict(model, loader, device)
    save_results(results, args.output_jsonl)


if __name__ == "__main__":
    main()
