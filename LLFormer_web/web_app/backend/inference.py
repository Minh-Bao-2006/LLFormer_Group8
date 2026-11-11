"""
inference.py — LLFormer Low-Light Enhancement Inference Engine
==============================================================
- 4 chế độ tăng sáng thân thiện cho người dùng cuối
- Khử hoàn toàn thuật ngữ chuyên sâu (PSNR, HSV, Lab, Joint, Loss...)
- Tối ưu tốc độ xử lý nhanh, mượt mà trên mọi thiết bị
"""

import sys
import os
import time
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

torch.set_num_threads(os.cpu_count() or 4)

MODEL_SRC = Path(__file__).parent.parent.parent / "llformer_src"
sys.path.insert(0, str(MODEL_SRC))
from model.LLFormer import LLFormer

MODEL_KWARGS = dict(
    inp_channels=3,
    out_channels=3,
    dim=16,
    num_blocks=[2, 4, 8, 16],
    num_refinement_blocks=2,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type="WithBias",
    attention=True,
    skip=False,
)
MUL = 16

# 4 chế độ thân thiện cho người dùng, sử dụng các checkpoint tối ưu nhất
CONFIG_SPECS = {
    "A": {
        "config": "A",
        "name": "Chế độ Tiêu chuẩn",
        "title": "Tiêu chuẩn (Tự nhiên)",
        "tag": "Tự nhiên",
        "desc": "Khôi phục ánh sáng cân bằng, giữ nguyên sắc độ tự nhiên của khung cảnh.",
        "feature": "Độ sáng tự nhiên",
        "folder": "A_baseline_s1",
        "color": "#9ca3af",
    },
    "B": {
        "config": "B",
        "name": "Chế độ Sống động",
        "title": "Sống động (Khuyên dùng)",
        "tag": "Khuyên dùng",
        "desc": "Tái tạo màu sắc rực rỡ, khử nhợt nhạt và phục hồi sắc thái tươi tắn nhất.",
        "feature": "Màu sắc tối ưu",
        "folder": "B_hsv_s1",
        "color": "#10b981",
    },
    "C": {
        "config": "C",
        "name": "Chế độ Hài hòa",
        "title": "Hài hòa (Dịu mắt)",
        "tag": "Dịu mắt",
        "desc": "Cân bằng dải sáng mềm mại, hạn chế chói gắt và bảo vệ chi tiết êm dịu.",
        "feature": "Ánh sáng êm dịu",
        "folder": "C_lab_s1",
        "color": "#3b82f6",
    },
    "D": {
        "config": "D",
        "name": "Chế độ Rõ nét",
        "title": "Rõ nét (Chi tiết cao)",
        "tag": "Chi tiết cao",
        "desc": "Tăng cường độ tương phản và làm nổi bật các chi tiết trong bóng tối sâu.",
        "feature": "Tương phản cao",
        "folder": "D_joint_s1",
        "color": "#f59e0b",
    },
}


def _reflect_pad(tensor: torch.Tensor, mul: int = MUL):
    _, _, h, w = tensor.shape
    ph = (mul - h % mul) % mul
    pw = (mul - w % mul) % mul
    if ph == 0 and pw == 0:
        return tensor, h, w
    return F.pad(tensor, (0, pw, 0, ph), mode="reflect"), h, w


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[LLFormer, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        state = ckpt
        meta = {}

    model = LLFormer(**MODEL_KWARGS)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model, meta


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray(np.rint(arr * 255.0).astype(np.uint8), mode="RGB")


def assess_image_quality(img: Image.Image) -> Dict:
    """
    Đánh giá chi tiết chất lượng ảnh đầu vào:
    - Độ sáng thực tế của các vùng
    - Tỷ lệ diện tích điểm ảnh theo từng mức ánh sáng
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    mask_dark = lum < 0.15
    mask_low  = (lum >= 0.15) & (lum < 0.40)
    mask_mid  = (lum >= 0.40) & (lum < 0.70)
    mask_high = lum >= 0.70

    dark_lum_val = float(lum[mask_dark].mean() * 100.0) if mask_dark.sum() > 0 else 0.0
    low_lum_val  = float(lum[mask_low].mean() * 100.0)  if mask_low.sum() > 0  else None
    mid_lum_val  = float(lum[mask_mid].mean() * 100.0)  if mask_mid.sum() > 0  else None
    high_lum_val = float(lum[mask_high].mean() * 100.0) if mask_high.sum() > 0 else None

    dark_area_pct = float(mask_dark.mean() * 100.0)
    low_area_pct  = float(mask_low.mean() * 100.0)
    mid_area_pct  = float(mask_mid.mean() * 100.0)
    high_area_pct = float(mask_high.mean() * 100.0)

    mean_lum   = float(lum.mean() * 100.0)
    median_lum = float(np.median(lum) * 100.0)
    min_lum    = float(lum.min() * 100.0)
    max_lum    = float(lum.max() * 100.0)

    p25, p75 = float(np.percentile(lum, 25)), float(np.percentile(lum, 75))
    contrast_iqr = round(p75 - p25, 4)
    noise_est = float(lum[mask_dark].std()) if mask_dark.sum() > 100 else float(lum.std())

    if mean_lum < 10.0:
        darkness_level = "Rất tối (Cần tăng sáng mạnh)"
    elif mean_lum < 25.0:
        darkness_level = "Tối (Thiếu ánh sáng)"
    elif mean_lum < 40.0:
        darkness_level = "Thiếu sáng nhẹ"
    else:
        darkness_level = "Ánh sáng vừa"

    return {
        "width": w,
        "height": h,
        "megapixels": round(w * h / 1e6, 2),
        "mean_luminance": round(mean_lum, 1),
        "median_luminance": round(median_lum, 1),
        "min_luminance": round(min_lum, 1),
        "max_luminance": round(max_lum, 1),

        "dark_lum_val": round(dark_lum_val, 1),
        "low_lum_val": round(low_lum_val, 1) if low_lum_val is not None else None,
        "mid_lum_val": round(mid_lum_val, 1) if mid_lum_val is not None else None,
        "high_lum_val": round(high_lum_val, 1) if high_lum_val is not None else None,

        "dark_area_pct": round(dark_area_pct, 1),
        "low_area_pct": round(low_area_pct, 1),
        "mid_area_pct": round(mid_area_pct, 1),
        "high_area_pct": round(high_area_pct, 1),

        "contrast_iqr": contrast_iqr,
        "noise_estimate": round(noise_est, 4),
        "darkness_level": darkness_level,
    }


def compute_enhancement_metrics(before: Image.Image, after: Image.Image) -> Dict:
    """So sánh số liệu chuẩn xác trước và sau khi tăng sáng."""
    b_arr = np.asarray(before.convert("RGB"), dtype=np.float32) / 255.0
    a_arr = np.asarray(after.convert("RGB"),  dtype=np.float32) / 255.0

    b_lum = 0.299 * b_arr[..., 0] + 0.587 * b_arr[..., 1] + 0.114 * b_arr[..., 2]
    a_lum = 0.299 * a_arr[..., 0] + 0.587 * a_arr[..., 1] + 0.114 * a_arr[..., 2]

    before_mean = float(b_lum.mean())
    after_mean  = float(a_lum.mean())
    gain = after_mean - before_mean
    gain_pct = (gain / max(before_mean, 1e-5)) * 100.0

    dark_before = float((b_lum < 0.15).mean()) * 100.0
    dark_after  = float((a_lum < 0.15).mean()) * 100.0
    dark_reduction = dark_before - dark_after

    b_contrast = float(b_lum.std())
    a_contrast = float(a_lum.std())

    r_shift = float((a_arr[..., 0] - b_arr[..., 0]).mean())
    g_shift = float((a_arr[..., 1] - b_arr[..., 1]).mean())
    b_shift = float((a_arr[..., 2] - b_arr[..., 2]).mean())

    return {
        "before_mean_lum": round(before_mean * 100.0, 1),
        "after_mean_lum": round(after_mean * 100.0, 1),
        "brightness_gain": round(gain * 100.0, 1),
        "brightness_gain_pct": round(gain_pct, 1),
        "dark_before_pct": round(dark_before, 1),
        "dark_after_pct": round(dark_after, 1),
        "dark_reduction_pct": round(dark_reduction, 1),
        "contrast_before": round(b_contrast * 100.0, 1),
        "contrast_after": round(a_contrast * 100.0, 1),
        "r_shift": round(r_shift * 100.0, 1),
        "g_shift": round(g_shift * 100.0, 1),
        "b_shift": round(b_shift * 100.0, 1),
    }


@torch.no_grad()
def run_enhance(
    model: LLFormer,
    input_img: Image.Image,
    device: torch.device,
) -> Tuple[Image.Image, float]:
    inp = pil_to_tensor(input_img).to(device)
    padded, h_orig, w_orig = _reflect_pad(inp, MUL)

    t0 = time.perf_counter()
    amp_on = device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=amp_on):
        out = model(padded)[:, :, :h_orig, :w_orig].clamp(0.0, 1.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return tensor_to_pil(out), round(elapsed_ms, 1)
