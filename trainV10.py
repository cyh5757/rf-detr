#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RF-DETR Segmentation Training Script (trainV10)

핵심:
- trainV9 기반 타일 학습 파이프라인 유지
- train split에 offline augmentation 추가(타일 이미지 + segmentation 동기화)
- augmentation 검증용 preview 저장 옵션 지원

실행 예:
  cd /home/mbd1234/rf-detr
  python trainV10.py --size M --resolution 432 --tile-enable \
    --train-aug-preset aggressive \
    --aug-enable --aug-policy conservative
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import shutil
import sys
import weakref
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pycocotools.mask as coco_mask
except Exception:
    coco_mask = None


# -----------------------------
# 1) Size -> Resolution mapping
# -----------------------------
SIZE_TO_RESOLUTION = {
    "N": 312,
    "S": 384,
    "M": 432,
    "L": 504,
    "XL": 624,
    "2XL": 768,
}

SIZE_TO_PATCH_SIZE = {
    "N": 12,
    "S": 12,
    "M": 12,
    "L": 12,
    "XL": 12,
    "2XL": 12,
}

SIZE_TO_NUM_WINDOWS = {
    "N": 1,
    "S": 2,
    "M": 2,
    "L": 2,
    "XL": 2,
    "2XL": 2,
}


def validate_resolution(size: str, resolution: int) -> tuple[int, int, int]:
    patch_size = SIZE_TO_PATCH_SIZE[size]
    num_windows = SIZE_TO_NUM_WINDOWS[size]
    block_size = patch_size * num_windows

    if resolution <= 0:
        raise ValueError(f"resolution은 양수여야 함: {resolution}")
    if resolution % patch_size != 0:
        lower = patch_size * (resolution // patch_size)
        upper = lower + patch_size
        raise ValueError(
            f"resolution={resolution}는 patch_size={patch_size}의 배수여야 함 "
            f"(예: {lower} 또는 {upper})."
        )
    if resolution % block_size != 0:
        lower = block_size * (resolution // block_size)
        upper = lower + block_size
        raise ValueError(
            f"resolution={resolution}는 size={size}에서 patch_size({patch_size})*num_windows({num_windows})="
            f"{block_size}의 배수여야 함 (예: {lower} 또는 {upper})."
        )
    return patch_size, num_windows, resolution // patch_size


# -----------------------------
# 2) 유틸
# -----------------------------
def create_output_dir(base_dir: str | Path, prefix: str) -> Path:
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    out_dir = base_dir / f"{timestamp}__{prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def cleanup_gpu_memory(obj=None, verbose: bool = False) -> None:
    if not torch.cuda.is_available():
        if verbose:
            print("[INFO] CUDA is not available. No GPU cleanup needed.")
        return

    def get_memory_stats():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return allocated, reserved

    torch.cuda.synchronize()

    if verbose:
        alloc, reserv = get_memory_stats()
        print(
            f"[Before] Allocated: {alloc / 1024**2:.2f} MB | "
            f"Reserved: {reserv / 1024**2:.2f} MB"
        )

    if obj is not None:
        ref = weakref.ref(obj)
        del obj
        if ref() is not None and verbose:
            print("[WARNING] Object not fully garbage collected yet.")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    if verbose:
        alloc, reserv = get_memory_stats()
        print(
            f"[After]  Allocated: {alloc / 1024**2:.2f} MB | "
            f"Reserved: {reserv / 1024**2:.2f} MB"
        )


def assert_coco_layout(data_root: Path) -> tuple[Path, Path]:
    train_ann = data_root / "train" / "_annotations.coco.json"
    valid_ann = data_root / "valid" / "_annotations.coco.json"
    if not train_ann.exists():
        raise FileNotFoundError(f"없음: {train_ann}")
    if not valid_ann.exists():
        alt = data_root / "val" / "_annotations.coco.json"
        if alt.exists():
            valid_ann = alt
        else:
            raise FileNotFoundError(f"없음: {valid_ann} (또는 {alt})")
    return train_ann, valid_ann


def quick_segmentation_sanity_check(coco_json: Path, sample_n: int = 100) -> None:
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    anns = coco.get("annotations", [])
    if not anns:
        raise ValueError(f"annotations 비어있음: {coco_json}")

    random.shuffle(anns)
    checked = 0
    ok = 0

    for ann in anns:
        seg = ann.get("segmentation", None)
        checked += 1

        valid = False
        if isinstance(seg, list) and len(seg) > 0:
            if all(isinstance(x, (int, float)) for x in seg):
                valid = (len(seg) >= 6 and len(seg) % 2 == 0)
            elif all(isinstance(poly, list) for poly in seg):
                valid = any((len(poly) >= 6 and len(poly) % 2 == 0) for poly in seg)

        if valid:
            ok += 1

        if checked >= sample_n:
            break

    if ok == 0:
        raise ValueError(
            "샘플에서 유효한 segmentation polygon을 하나도 못 찾음. "
            "COCO segmentation 필드/라벨링 상태를 확인해봐."
        )

    print(f"[SANITY] segmentation polygons found: {ok}/{checked} (sample)")


def write_run_meta(output_dir: Path, meta: dict) -> None:
    p = output_dir / "run_meta.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_tiling_dependencies() -> None:
    if cv2 is None:
        raise ImportError(
            "trainV10 tile mode requires OpenCV (`cv2`). "
            "Install it in the training environment (e.g., pip install opencv-python)."
        )
    if coco_mask is None:
        raise ImportError(
            "trainV10 tile mode requires `pycocotools`. "
            "Install it in the training environment (e.g., pip install pycocotools)."
        )


def get_aug_profile(policy: str) -> dict[str, float]:
    p = str(policy).strip().lower()
    if p == "aggressive":
        return {
            "flip_lr": 0.50,
            "flip_ud": 0.30,
            "rot90": 0.35,
            "brightness_contrast": 0.70,
            "blur": 0.30,
            "noise": 0.30,
            "alpha_min": 0.75,
            "alpha_max": 1.25,
            "beta_min": -35.0,
            "beta_max": 35.0,
            "noise_sigma": 16.0,
        }
    return {
        "flip_lr": 0.50,
        "flip_ud": 0.10,
        "rot90": 0.15,
        "brightness_contrast": 0.50,
        "blur": 0.15,
        "noise": 0.15,
        "alpha_min": 0.90,
        "alpha_max": 1.10,
        "beta_min": -15.0,
        "beta_max": 15.0,
        "noise_sigma": 8.0,
    }


TRAIN_AUG_PRESET_CHOICES = [
    "default",
    "none",
    "conservative",
    "aggressive",
    "aerial",
    "industrial",
]


def load_train_aug_config_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"train_aug_config_json 읽기 실패: {path} ({e})") from e

    if not isinstance(payload, dict):
        raise ValueError(
            f"train_aug_config_json는 dict(JSON object)여야 함: {path}"
        )
    for k, v in payload.items():
        if not isinstance(k, str):
            raise ValueError(f"aug_config key는 문자열이어야 함: key={k!r}")
        if not isinstance(v, dict):
            raise ValueError(f"aug_config[{k!r}] 값은 dict여야 함: {type(v).__name__}")
    return payload


def resolve_train_aug_config(train_aug_preset: str, train_aug_config_json: Optional[Path]) -> tuple[Optional[dict], str]:
    if train_aug_config_json is not None:
        return load_train_aug_config_json(train_aug_config_json), f"custom:{train_aug_config_json}"

    preset = str(train_aug_preset).strip().lower()
    if preset == "default":
        return None, "default"
    if preset == "none":
        return {}, "none"

    try:
        from rfdetr.datasets.aug_config import (
            AUG_AERIAL,
            AUG_AGGRESSIVE,
            AUG_CONSERVATIVE,
            AUG_INDUSTRIAL,
        )
    except Exception as e:
        raise RuntimeError(
            "rfdetr.datasets.aug_config import 실패. "
            "train augmentation preset을 사용하려면 rfdetr import가 가능해야 합니다."
        ) from e

    presets = {
        "conservative": AUG_CONSERVATIVE,
        "aggressive": AUG_AGGRESSIVE,
        "aerial": AUG_AERIAL,
        "industrial": AUG_INDUSTRIAL,
    }
    if preset not in presets:
        raise ValueError(
            f"지원하지 않는 train_aug_preset: {train_aug_preset}. "
            f"choices={TRAIN_AUG_PRESET_CHOICES}"
        )
    return copy.deepcopy(presets[preset]), preset


def draw_tile_annotations(image_bgr: np.ndarray, anns: list[dict]) -> np.ndarray:
    vis = image_bgr.copy()
    for ann in anns:
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = max(0, int(round(x + w)))
        y2 = max(0, int(round(y + h)))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (20, 230, 20), 1)

        for poly in parse_polygon_segmentation(ann.get("segmentation")):
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 3:
                continue
            pts_i = np.round(pts).astype(np.int32)
            cv2.polylines(vis, [pts_i], isClosed=True, color=(20, 180, 255), thickness=1)
    return vis


def ann_to_tile_mask(ann: dict, tile_size: int) -> np.ndarray:
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    polygons = parse_polygon_segmentation(ann.get("segmentation"))
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        pts_i = np.round(pts).astype(np.int32)
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, tile_size - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, tile_size - 1)
        cv2.fillPoly(mask, [pts_i], 1)
    return mask


def mask_to_tile_ann(mask: np.ndarray, category_id: int, iscrowd: int = 0) -> Optional[dict]:
    m = (mask > 0).astype(np.uint8)
    area = float(m.sum())
    if area <= 0:
        return None

    contours, _ = cv2.findContours(m * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segs: list[list[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        pts = contour.reshape(-1, 2).astype(np.float32)
        poly = pts.flatten().tolist()
        if len(poly) >= 6:
            segs.append([float(v) for v in poly])
    if not segs:
        return None

    ys, xs = np.where(m > 0)
    if ys.size == 0 or xs.size == 0:
        return None

    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw < 1.0 or bh < 1.0:
        return None

    return {
        "category_id": int(category_id),
        "bbox": [x1, y1, bw, bh],
        "area": area,
        "segmentation": segs,
        "iscrowd": int(iscrowd),
    }


def augment_tile_image_and_anns(
    image_bgr: np.ndarray,
    anns: list[dict],
    tile_size: int,
    policy: str,
    rng: random.Random,
) -> tuple[np.ndarray, list[dict], list[str]]:
    cfg = get_aug_profile(policy)
    ops: list[str] = []

    masks = [ann_to_tile_mask(ann, tile_size) for ann in anns]
    img = image_bgr.copy()

    if rng.random() < cfg["flip_lr"]:
        img = np.ascontiguousarray(np.flip(img, axis=1))
        masks = [np.ascontiguousarray(np.flip(m, axis=1)) for m in masks]
        ops.append("flip_lr")

    if rng.random() < cfg["flip_ud"]:
        img = np.ascontiguousarray(np.flip(img, axis=0))
        masks = [np.ascontiguousarray(np.flip(m, axis=0)) for m in masks]
        ops.append("flip_ud")

    if rng.random() < cfg["rot90"]:
        k = rng.choice([1, 2, 3])
        img = np.ascontiguousarray(np.rot90(img, k))
        masks = [np.ascontiguousarray(np.rot90(m, k)) for m in masks]
        ops.append(f"rot90_k{k}")

    if rng.random() < cfg["brightness_contrast"]:
        alpha = rng.uniform(cfg["alpha_min"], cfg["alpha_max"])
        beta = rng.uniform(cfg["beta_min"], cfg["beta_max"])
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        ops.append("brightness_contrast")

    if rng.random() < cfg["blur"]:
        ksize = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)
        ops.append("blur")

    if rng.random() < cfg["noise"]:
        sigma = cfg["noise_sigma"]
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        n = np_rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
        ops.append("noise")

    aug_anns: list[dict] = []
    for ann, mask in zip(anns, masks):
        rebuilt = mask_to_tile_ann(mask, category_id=int(ann["category_id"]), iscrowd=int(ann.get("iscrowd", 0)))
        if rebuilt is not None:
            aug_anns.append(rebuilt)

    return img, aug_anns, ops


def summarize_counts(values: list[int]) -> dict:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }

    sv = sorted(values)

    def pick(q: float) -> int:
        i = max(0, min(len(sv) - 1, int((len(sv) - 1) * q)))
        return int(sv[i])

    mid = len(sv) // 2
    if len(sv) % 2 == 0:
        median = (sv[mid - 1] + sv[mid]) / 2.0
    else:
        median = float(sv[mid])

    return {
        "n": int(len(sv)),
        "mean": float(sum(sv) / len(sv)),
        "median": float(median),
        "p90": pick(0.90),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": int(sv[-1]),
    }


# -----------------------------
# 3) 타일 데이터셋 생성
# -----------------------------
def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def bbox_intersects_tile(bbox: list[float], tile_x: int, tile_y: int, tile_size: int) -> bool:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    return not (
        x + w <= tile_x
        or y + h <= tile_y
        or x >= tile_x + tile_size
        or y >= tile_y + tile_size
    )


def parse_polygon_segmentation(segmentation) -> list[list[float]]:
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return []

    if all(isinstance(v, (int, float)) for v in segmentation):
        if len(segmentation) >= 6 and len(segmentation) % 2 == 0:
            return [[float(v) for v in segmentation]]
        return []

    out = []
    for poly in segmentation:
        if not isinstance(poly, list):
            continue
        if len(poly) < 6 or len(poly) % 2 != 0:
            continue
        out.append([float(v) for v in poly])
    return out


def polygons_to_roi_mask(polygons: list[list[float]], roi_x: int, roi_y: int, roi_w: int, roi_h: int) -> np.ndarray:
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        pts[:, 0] -= float(roi_x)
        pts[:, 1] -= float(roi_y)
        pts_i = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts_i], 1)
    return mask


def decode_rle_to_mask(segmentation, image_h: int, image_w: int) -> Optional[np.ndarray]:
    try:
        if isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
            rle = coco_mask.frPyObjects(segmentation, image_h, image_w)
        else:
            rle = segmentation
        m = coco_mask.decode(rle)
        if m.ndim == 3:
            m = np.any(m, axis=2)
        return m.astype(np.uint8)
    except Exception:
        return None


def clip_annotation_to_tile(
    ann: dict,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    image_h: int,
    image_w: int,
    min_area: float,
    min_visible_ratio: float,
) -> Optional[dict]:
    bbox = ann.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    bbox = [float(v) for v in bbox]
    if not bbox_intersects_tile(bbox, tile_x, tile_y, tile_size):
        return None

    bx, by, bw, bh = bbox
    ix1 = max(bx, float(tile_x))
    iy1 = max(by, float(tile_y))
    ix2 = min(bx + bw, float(tile_x + tile_size))
    iy2 = min(by + bh, float(tile_y + tile_size))
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    rx1 = max(0, int(math.floor(ix1)))
    ry1 = max(0, int(math.floor(iy1)))
    rx2 = min(image_w, int(math.ceil(ix2)))
    ry2 = min(image_h, int(math.ceil(iy2)))
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi_w = rx2 - rx1
    roi_h = ry2 - ry1

    seg = ann.get("segmentation")
    polygons = parse_polygon_segmentation(seg)

    if polygons:
        roi_mask = polygons_to_roi_mask(polygons, rx1, ry1, roi_w, roi_h)
    else:
        full_mask = decode_rle_to_mask(seg, image_h, image_w)
        if full_mask is None:
            return None
        roi_mask = full_mask[ry1:ry2, rx1:rx2].astype(np.uint8)

    cx1 = max(0, tile_x - rx1)
    cy1 = max(0, tile_y - ry1)
    cx2 = min(roi_w, tile_x + tile_size - rx1)
    cy2 = min(roi_h, tile_y + tile_size - ry1)
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    clipped = (roi_mask[cy1:cy2, cx1:cx2] > 0).astype(np.uint8)
    area = float(clipped.sum())
    if area < float(min_area):
        return None

    orig_area = float(ann.get("area", 0.0))
    if orig_area <= 0:
        orig_area = max(area, 1.0)
    visible_ratio = area / max(orig_area, 1e-6)
    if visible_ratio < float(min_visible_ratio):
        return None

    contours, _ = cv2.findContours(clipped * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tile_segs: list[list[float]] = []

    ox = float(rx1 + cx1 - tile_x)
    oy = float(ry1 + cy1 - tile_y)

    for contour in contours:
        if contour.shape[0] < 3:
            continue
        pts = contour.reshape(-1, 2).astype(np.float32)
        pts[:, 0] += ox
        pts[:, 1] += oy
        poly = pts.flatten().tolist()
        if len(poly) >= 6:
            tile_segs.append([float(v) for v in poly])

    if not tile_segs:
        return None

    ys, xs = np.where(clipped > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    x1 = float(xs.min() + ox)
    y1 = float(ys.min() + oy)
    x2 = float(xs.max() + 1 + ox)
    y2 = float(ys.max() + 1 + oy)

    bw_local = max(0.0, x2 - x1)
    bh_local = max(0.0, y2 - y1)
    if bw_local < 1.0 or bh_local < 1.0:
        return None

    return {
        "category_id": int(ann.get("category_id", 1)),
        "bbox": [x1, y1, bw_local, bh_local],
        "area": area,
        "segmentation": tile_segs,
        "iscrowd": int(ann.get("iscrowd", 0)),
    }


def extract_tile_with_padding(image_bgr: np.ndarray, tile_x: int, tile_y: int, tile_size: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    tile = np.zeros((tile_size, tile_size, 3), dtype=image_bgr.dtype)

    sx1, sy1 = tile_x, tile_y
    sx2 = min(w, tile_x + tile_size)
    sy2 = min(h, tile_y + tile_size)

    crop = image_bgr[sy1:sy2, sx1:sx2]
    tile[: crop.shape[0], : crop.shape[1]] = crop
    return tile


def build_tiled_split(
    split_name: str,
    src_split_dir: Path,
    src_ann_path: Path,
    dst_split_dir: Path,
    tile_size: int,
    stride: int,
    min_area: float,
    min_visible_ratio: float,
    max_empty_ratio: float,
    augment_train: bool = False,
    aug_policy: str = "conservative",
    aug_train_copies: int = 1,
    aug_verify_count: int = 0,
    aug_verify_dir: Optional[Path] = None,
    seed: int = 42,
) -> dict:
    coco = json.loads(src_ann_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[int(ann.get("image_id"))].append(ann)

    dst_split_dir.mkdir(parents=True, exist_ok=True)

    out_images: list[dict] = []
    out_annotations: list[dict] = []

    next_img_id = 1
    next_ann_id = 1

    total_tiles = 0
    kept_tiles = 0
    kept_positive_tiles = 0
    kept_empty_tiles = 0
    missing_images = 0

    ann_per_tile: list[int] = []
    augmented_tiles = 0
    augmented_annotations = 0
    aug_verify_saved = 0
    aug_ops_count: dict[str, int] = defaultdict(int)
    aug_enabled_here = bool(augment_train and split_name == "train" and aug_train_copies > 0)

    if aug_enabled_here and aug_verify_dir is not None and aug_verify_count > 0:
        aug_verify_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed + (0 if split_name == "train" else 1_000_000))

    def append_tile_record(tile_img: np.ndarray, tile_anns: list[dict], tile_name: str) -> None:
        nonlocal next_img_id, next_ann_id, kept_tiles, kept_positive_tiles, kept_empty_tiles
        cv2.imwrite(str(dst_split_dir / tile_name), tile_img)

        out_images.append(
            {
                "id": next_img_id,
                "file_name": tile_name,
                "width": tile_size,
                "height": tile_size,
            }
        )

        for ta in tile_anns:
            out_annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": next_img_id,
                    "category_id": int(ta["category_id"]),
                    "bbox": [float(v) for v in ta["bbox"]],
                    "area": float(ta["area"]),
                    "segmentation": ta["segmentation"],
                    "iscrowd": int(ta["iscrowd"]),
                }
            )
            next_ann_id += 1

        if tile_anns:
            kept_positive_tiles += 1
        else:
            kept_empty_tiles += 1

        ann_per_tile.append(len(tile_anns))
        kept_tiles += 1
        next_img_id += 1

    for idx, image_info in enumerate(images, start=1):
        image_id = int(image_info.get("id"))
        file_name = str(image_info.get("file_name"))

        src_img_path = src_split_dir / file_name
        if not src_img_path.exists():
            fallback = src_split_dir / Path(file_name).name
            if fallback.exists():
                src_img_path = fallback
            else:
                missing_images += 1
                continue

        image = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
        if image is None:
            missing_images += 1
            continue

        h, w = image.shape[:2]
        xs = tile_starts(w, tile_size, stride)
        ys = tile_starts(h, tile_size, stride)

        stem = str(Path(file_name).with_suffix("")).replace("/", "_").replace("\\", "_")
        ext = Path(file_name).suffix
        if not ext:
            ext = ".jpg"

        img_anns = anns_by_image.get(image_id, [])

        for y in ys:
            for x in xs:
                total_tiles += 1

                tile_anns: list[dict] = []
                for ann in img_anns:
                    if int(ann.get("iscrowd", 0)) != 0:
                        continue
                    clipped = clip_annotation_to_tile(
                        ann=ann,
                        tile_x=x,
                        tile_y=y,
                        tile_size=tile_size,
                        image_h=h,
                        image_w=w,
                        min_area=min_area,
                        min_visible_ratio=min_visible_ratio,
                    )
                    if clipped is not None:
                        tile_anns.append(clipped)

                keep_tile = len(tile_anns) > 0
                if not keep_tile and max_empty_ratio > 0:
                    allowed_empty = int(kept_positive_tiles * max_empty_ratio)
                    if kept_empty_tiles < allowed_empty:
                        keep_tile = True

                if not keep_tile:
                    continue

                tile_img = extract_tile_with_padding(image, tile_x=x, tile_y=y, tile_size=tile_size)
                tile_name = f"{stem}__id{image_id}__x{x}_y{y}_s{tile_size}{ext}"
                append_tile_record(tile_img=tile_img, tile_anns=tile_anns, tile_name=tile_name)

                if aug_enabled_here and len(tile_anns) > 0:
                    for aug_idx in range(max(0, int(aug_train_copies))):
                        aug_img, aug_anns, aug_ops = augment_tile_image_and_anns(
                            image_bgr=tile_img,
                            anns=tile_anns,
                            tile_size=tile_size,
                            policy=aug_policy,
                            rng=rng,
                        )
                        if len(aug_anns) == 0 and len(tile_anns) > 0:
                            continue

                        aug_name = f"{stem}__id{image_id}__x{x}_y{y}_s{tile_size}__aug{aug_idx + 1}{ext}"
                        append_tile_record(tile_img=aug_img, tile_anns=aug_anns, tile_name=aug_name)
                        augmented_tiles += 1
                        augmented_annotations += len(aug_anns)
                        for op in aug_ops:
                            aug_ops_count[op] += 1

                        if aug_verify_dir is not None and aug_verify_saved < aug_verify_count:
                            vis_orig = draw_tile_annotations(tile_img, tile_anns)
                            vis_aug = draw_tile_annotations(aug_img, aug_anns)
                            pair = np.concatenate([vis_orig, vis_aug], axis=1)
                            preview_name = f"{stem}__id{image_id}__x{x}_y{y}__aug{aug_idx + 1}.jpg"
                            cv2.imwrite(str(aug_verify_dir / preview_name), pair)
                            meta = {
                                "source_tile": tile_name,
                                "aug_tile": aug_name,
                                "ops": aug_ops,
                                "src_ann_count": len(tile_anns),
                                "aug_ann_count": len(aug_anns),
                            }
                            (aug_verify_dir / f"{Path(preview_name).stem}.json").write_text(
                                json.dumps(meta, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                            aug_verify_saved += 1

        if idx % 20 == 0 or idx == len(images):
            print(
                f"[TILE][{split_name}] {idx}/{len(images)} images | "
                f"kept_tiles={kept_tiles} anns={len(out_annotations)}"
            )

    out_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": out_images,
        "annotations": out_annotations,
        "categories": coco.get("categories", []),
    }

    out_ann_path = dst_split_dir / "_annotations.coco.json"
    out_ann_path.write_text(json.dumps(out_coco, ensure_ascii=False), encoding="utf-8")

    stats = {
        "split": split_name,
        "source_image_count": len(images),
        "source_annotation_count": len(annotations),
        "missing_images": missing_images,
        "tile_size": int(tile_size),
        "stride": int(stride),
        "total_candidate_tiles": int(total_tiles),
        "kept_tiles": int(kept_tiles),
        "kept_positive_tiles": int(kept_positive_tiles),
        "kept_empty_tiles": int(kept_empty_tiles),
        "output_image_count": len(out_images),
        "output_annotation_count": len(out_annotations),
        "ann_per_tile": summarize_counts(ann_per_tile),
        "augmentation_enabled": bool(aug_enabled_here),
        "augmentation_policy": aug_policy if aug_enabled_here else None,
        "augmentation_copies_per_tile": int(aug_train_copies) if aug_enabled_here else 0,
        "augmented_tiles": int(augmented_tiles),
        "augmented_annotations": int(augmented_annotations),
        "augmentation_ops_count": {k: int(v) for k, v in sorted(aug_ops_count.items())},
        "augmentation_verify_saved": int(aug_verify_saved),
        "augmentation_verify_dir": str(aug_verify_dir) if aug_verify_dir is not None and aug_enabled_here else None,
        "output_ann_path": str(out_ann_path),
    }
    return stats


def default_tile_output_root(
    data_root: Path,
    tile_size: int,
    tile_overlap: float,
    aug_enable: bool,
    aug_policy: str,
    aug_train_copies: int,
) -> Path:
    ov = int(round(tile_overlap * 100))
    suffix = f"{data_root.name}_tiled_r{tile_size}_ov{ov:02d}"
    if aug_enable:
        suffix += f"__aug_{aug_policy}_x{max(1, int(aug_train_copies))}"
    return data_root.parent / suffix


def ensure_tiled_dataset(
    data_root: Path,
    tile_output_root: Path,
    tile_size: int,
    tile_overlap: float,
    tile_min_area: float,
    tile_min_visible_ratio: float,
    tile_max_empty_ratio: float,
    tile_rebuild: bool,
    aug_enable: bool,
    aug_policy: str,
    aug_train_copies: int,
    aug_verify_count: int,
    aug_verify_dir: Optional[Path],
    seed: int,
) -> tuple[Path, dict]:
    train_ann, valid_ann = assert_coco_layout(data_root)

    out_root = tile_output_root
    out_train_ann = out_root / "train" / "_annotations.coco.json"
    out_valid_ann = out_root / "valid" / "_annotations.coco.json"
    meta_path = out_root / "tile_meta.json"

    if out_train_ann.exists() and out_valid_ann.exists() and not tile_rebuild:
        print(f"[INFO] Reusing tiled dataset: {out_root}")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        return out_root, meta

    if tile_rebuild and out_root.exists():
        print(f"[INFO] Removing previous tiled dataset: {out_root}")
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(round(tile_size * (1.0 - tile_overlap))))

    print("[INFO] Building tiled dataset...")
    print(f"[INFO] source data_root    : {data_root}")
    print(f"[INFO] tile output_root   : {out_root}")
    print(f"[INFO] tile_size/overlap  : {tile_size}/{tile_overlap}")
    print(f"[INFO] stride             : {stride}")
    print(f"[INFO] min_area           : {tile_min_area}")
    print(f"[INFO] min_visible_ratio  : {tile_min_visible_ratio}")
    print(f"[INFO] max_empty_ratio    : {tile_max_empty_ratio}")
    print(f"[INFO] aug_enable         : {aug_enable}")
    if aug_enable:
        print(f"[INFO] aug_policy         : {aug_policy}")
        print(f"[INFO] aug_train_copies   : {aug_train_copies}")
        print(f"[INFO] aug_verify_count   : {aug_verify_count}")
        if aug_verify_dir is not None:
            print(f"[INFO] aug_verify_dir     : {aug_verify_dir}")

    train_stats = build_tiled_split(
        split_name="train",
        src_split_dir=train_ann.parent,
        src_ann_path=train_ann,
        dst_split_dir=out_root / "train",
        tile_size=tile_size,
        stride=stride,
        min_area=tile_min_area,
        min_visible_ratio=tile_min_visible_ratio,
        max_empty_ratio=tile_max_empty_ratio,
        augment_train=aug_enable,
        aug_policy=aug_policy,
        aug_train_copies=aug_train_copies,
        aug_verify_count=aug_verify_count,
        aug_verify_dir=aug_verify_dir,
        seed=seed,
    )

    valid_stats = build_tiled_split(
        split_name="valid",
        src_split_dir=valid_ann.parent,
        src_ann_path=valid_ann,
        dst_split_dir=out_root / "valid",
        tile_size=tile_size,
        stride=stride,
        min_area=tile_min_area,
        min_visible_ratio=tile_min_visible_ratio,
        max_empty_ratio=0.0,
        augment_train=False,
        aug_policy=aug_policy,
        aug_train_copies=0,
        aug_verify_count=0,
        aug_verify_dir=None,
        seed=seed,
    )

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_data_root": str(data_root),
        "tile_output_root": str(out_root),
        "tile_size": int(tile_size),
        "tile_overlap": float(tile_overlap),
        "stride": int(stride),
        "tile_min_area": float(tile_min_area),
        "tile_min_visible_ratio": float(tile_min_visible_ratio),
        "tile_max_empty_ratio_train": float(tile_max_empty_ratio),
        "augmentation": {
            "enabled": bool(aug_enable),
            "policy": aug_policy if aug_enable else None,
            "train_copies": int(aug_train_copies) if aug_enable else 0,
            "verify_count": int(aug_verify_count) if aug_enable else 0,
            "verify_dir": str(aug_verify_dir) if (aug_enable and aug_verify_dir is not None) else None,
        },
        "train": train_stats,
        "valid": valid_stats,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Tiled dataset build done.")
    print(f"[INFO] tile meta: {meta_path}")
    return out_root, meta


# -----------------------------
# 4) 모델 생성 + train 호출
# -----------------------------
def build_model(
    size: str,
    num_classes: int,
    class_names: list[str],
    resolution: int,
    amp: bool,
    gradient_checkpointing: bool,
):
    import rfdetr

    size = size.upper()
    patch_size, _, positional_encoding_size = validate_resolution(size, resolution)

    candidates_by_size = {
        "N": ["RFDETRSegNano"],
        "S": ["RFDETRSegSmall"],
        "M": ["RFDETRSegMedium"],
        "L": ["RFDETRSegLarge"],
        "XL": ["RFDETRSegXLarge"],
        "2XL": ["RFDETRSeg2XLarge"],
    }

    candidates = candidates_by_size.get(size)
    if not candidates:
        raise ValueError(f"지원하지 않는 size: {size} (N/S/M/L/XL/2XL)")

    last_err = None
    for cls_name in candidates:
        cls = getattr(rfdetr, cls_name, None)
        if cls is None:
            last_err = RuntimeError(f"Class not found in rfdetr: {cls_name}")
            continue

        try:
            kwargs = {
                "num_classes": num_classes,
                "class_names": class_names,
                "resolution": int(resolution),
                "amp": bool(amp),
                "gradient_checkpointing": bool(gradient_checkpointing),
                "positional_encoding_size": int(positional_encoding_size),
                "patch_size": int(patch_size),
            }
            return cls(**kwargs)
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"모델 클래스를 찾았지만 생성에 실패. candidates={candidates}, last_err={last_err}"
    )


def call_train(model, **kwargs):
    from rfdetr.config import ModelConfig, SegmentationTrainConfig

    allowed = set(SegmentationTrainConfig.model_fields.keys()) | set(ModelConfig.model_fields.keys())
    passthrough = {"aug_config"}
    allowed_effective = allowed | passthrough
    filtered = {k: v for k, v in kwargs.items() if k in allowed_effective and v is not None}
    dropped = [k for k, v in kwargs.items() if k not in allowed_effective and v is not None]
    if dropped:
        print(f"[WARN] Dropped unsupported train kwargs: {sorted(dropped)}")
    return model.train(**filtered)


# -----------------------------
# 5) Args
# -----------------------------
@dataclass
class TrainConfig:
    data_root: Path
    outputs_root: Path
    size: str
    resolution: int
    epochs: int
    batch_size: int
    grad_accum_steps: int
    num_workers: int
    pad_to_square: bool
    lr: float
    amp: bool
    gradient_checkpointing: bool
    num_classes: int
    class_names: list[str]
    early_stopping: bool
    resume: Optional[str]
    seed: int
    sanity_check: bool
    wandb_project: Optional[str]
    wandb_run_name: Optional[str]
    wandb_disable: bool
    tile_enable: bool
    tile_size: int
    tile_overlap: float
    tile_min_area: float
    tile_min_visible_ratio: float
    tile_max_empty_ratio: float
    tile_output_root: Path
    tile_rebuild: bool
    aug_enable: bool
    aug_policy: str
    aug_train_copies: int
    aug_verify_count: int
    aug_verify_dir: Optional[Path]
    aug_verify_only: bool
    train_aug_preset: str
    train_aug_config_json: Optional[Path]


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="RF-DETR Segmentation training (trainV10, tiled dataset + augmentation)")

    p.add_argument(
        "--data-root",
        type=str,
        default="/home/mbd1234/data/Optiresolve_result_total_20260223/total_data_refine_1024",
    )
    p.add_argument(
        "--outputs-root",
        type=str,
        default="/home/mbd1234/rf-detr/outputs",
        help="outputs 상위 루트. 실제 저장은 <outputs-root>/<dataset_tag>/...",
    )

    p.add_argument("--size", type=str, default="2XL", choices=list(SIZE_TO_RESOLUTION.keys()))
    p.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="미지정 시 size 기본값 사용",
    )

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pad-to-square", dest="pad_to_square", action="store_true", default=True)
    p.add_argument("--no-pad-to-square", dest="pad_to_square", action="store_false")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument(
        "--gradient-checkpointing",
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
    )

    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--class-names", type=str, nargs="+", default=["background", "cell"])
    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sanity-check", action="store_true")

    p.add_argument("--wandb-project", type=str, default="cell_rfdetr")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-disable", action="store_true")

    p.add_argument("--tile-enable", dest="tile_enable", action="store_true", default=True)
    p.add_argument("--no-tile-enable", dest="tile_enable", action="store_false")
    p.add_argument("--tile-size", type=int, default=None, help="미지정 시 resolution 사용")
    p.add_argument("--tile-overlap", type=float, default=0.20)
    p.add_argument("--tile-min-area", type=float, default=16.0)
    p.add_argument("--tile-min-visible-ratio", type=float, default=0.10)
    p.add_argument("--tile-max-empty-ratio", type=float, default=0.20)
    p.add_argument("--tile-output-root", type=str, default=None)
    p.add_argument("--tile-rebuild", action="store_true", help="기존 타일 데이터셋 강제 재생성")

    p.add_argument("--aug-enable", dest="aug_enable", action="store_true", default=False, help="타일 오프라인 증강 활성화")
    p.add_argument("--no-aug-enable", dest="aug_enable", action="store_false", help="타일 오프라인 증강 비활성화")
    p.add_argument(
        "--aug-policy",
        type=str,
        default="conservative",
        choices=["conservative", "aggressive"],
        help="타일 오프라인 증강 정책",
    )
    p.add_argument("--aug-train-copies", type=int, default=1, help="train 타일당 추가 오프라인 augmentation 샘플 수")
    p.add_argument("--aug-verify-count", type=int, default=20, help="오프라인 augmentation preview 저장 개수")
    p.add_argument("--aug-verify-dir", type=str, default=None, help="오프라인 augmentation preview 출력 폴더")
    p.add_argument("--aug-verify-only", action="store_true", help="타일/오프라인 증강 생성+검증까지만 수행하고 학습은 생략")

    p.add_argument(
        "--train-aug-preset",
        type=str,
        default="default",
        choices=TRAIN_AUG_PRESET_CHOICES,
        help=(
            "model.train(aug_config=...) 프리셋. "
            "default는 rfdetr 기본 AUG_CONFIG, none은 augmentation 비활성화({})."
        ),
    )
    p.add_argument(
        "--train-aug-config-json",
        type=str,
        default=None,
        help="custom aug_config JSON 파일 경로. 지정 시 --train-aug-preset보다 우선.",
    )

    a = p.parse_args()

    size = a.size.upper()
    resolution = a.resolution if a.resolution is not None else SIZE_TO_RESOLUTION[size]
    tile_size = a.tile_size if a.tile_size is not None else resolution

    if tile_size <= 0:
        raise ValueError(f"tile_size는 양수여야 함: {tile_size}")
    if not (0.0 <= a.tile_overlap < 1.0):
        raise ValueError(f"tile_overlap은 [0,1) 범위여야 함: {a.tile_overlap}")
    if a.tile_min_visible_ratio < 0.0 or a.tile_min_visible_ratio > 1.0:
        raise ValueError(f"tile_min_visible_ratio는 [0,1] 범위여야 함: {a.tile_min_visible_ratio}")
    if a.tile_max_empty_ratio < 0.0:
        raise ValueError(f"tile_max_empty_ratio는 0 이상이어야 함: {a.tile_max_empty_ratio}")
    if a.aug_train_copies < 0:
        raise ValueError(f"aug_train_copies는 0 이상이어야 함: {a.aug_train_copies}")
    if a.aug_verify_count < 0:
        raise ValueError(f"aug_verify_count는 0 이상이어야 함: {a.aug_verify_count}")
    if a.train_aug_config_json is not None:
        pth = Path(a.train_aug_config_json)
        if not pth.exists():
            raise FileNotFoundError(f"없음: train_aug_config_json={pth}")
        if not pth.is_file():
            raise ValueError(f"파일이 아님: train_aug_config_json={pth}")

    data_root = Path(a.data_root)
    tile_output_root = (
        Path(a.tile_output_root)
        if a.tile_output_root is not None
        else default_tile_output_root(
            data_root,
            tile_size=tile_size,
            tile_overlap=a.tile_overlap,
            aug_enable=bool(a.aug_enable),
            aug_policy=str(a.aug_policy),
            aug_train_copies=int(a.aug_train_copies),
        )
    )

    aug_verify_dir = Path(a.aug_verify_dir) if a.aug_verify_dir is not None else None
    train_aug_config_json = Path(a.train_aug_config_json) if a.train_aug_config_json is not None else None

    return TrainConfig(
        data_root=data_root,
        outputs_root=Path(a.outputs_root),
        size=size,
        resolution=resolution,
        epochs=a.epochs,
        batch_size=a.batch_size,
        grad_accum_steps=a.grad_accum_steps,
        num_workers=int(a.num_workers),
        pad_to_square=bool(a.pad_to_square),
        lr=a.lr,
        amp=bool(a.amp),
        gradient_checkpointing=bool(a.gradient_checkpointing),
        num_classes=a.num_classes,
        class_names=a.class_names,
        early_stopping=bool(a.early_stopping),
        resume=a.resume,
        seed=a.seed,
        sanity_check=bool(a.sanity_check),
        wandb_project=None if a.wandb_disable else a.wandb_project,
        wandb_run_name=a.wandb_run_name,
        wandb_disable=bool(a.wandb_disable),
        tile_enable=bool(a.tile_enable),
        tile_size=int(tile_size),
        tile_overlap=float(a.tile_overlap),
        tile_min_area=float(a.tile_min_area),
        tile_min_visible_ratio=float(a.tile_min_visible_ratio),
        tile_max_empty_ratio=float(a.tile_max_empty_ratio),
        tile_output_root=tile_output_root,
        tile_rebuild=bool(a.tile_rebuild),
        aug_enable=bool(a.aug_enable),
        aug_policy=str(a.aug_policy),
        aug_train_copies=int(a.aug_train_copies),
        aug_verify_count=int(a.aug_verify_count),
        aug_verify_dir=aug_verify_dir,
        aug_verify_only=bool(a.aug_verify_only),
        train_aug_preset=str(a.train_aug_preset),
        train_aug_config_json=train_aug_config_json,
    )


# -----------------------------
# 6) Main
# -----------------------------
def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    if cfg.aug_enable and not cfg.tile_enable:
        raise ValueError("aug_enable을 쓰려면 tile_enable도 True여야 합니다.")
    if cfg.aug_verify_only and not cfg.tile_enable:
        raise ValueError("aug_verify_only를 쓰려면 tile_enable도 True여야 합니다.")

    patch_size, num_windows, positional_encoding_size = validate_resolution(cfg.size, cfg.resolution)

    train_data_root = cfg.data_root
    tile_meta = None

    aug_verify_dir = cfg.aug_verify_dir
    if cfg.aug_enable and cfg.aug_verify_count > 0 and aug_verify_dir is None:
        aug_verify_dir = cfg.tile_output_root / "aug_verify"

    if cfg.tile_enable:
        ensure_tiling_dependencies()
        train_data_root, tile_meta = ensure_tiled_dataset(
            data_root=cfg.data_root,
            tile_output_root=cfg.tile_output_root,
            tile_size=cfg.tile_size,
            tile_overlap=cfg.tile_overlap,
            tile_min_area=cfg.tile_min_area,
            tile_min_visible_ratio=cfg.tile_min_visible_ratio,
            tile_max_empty_ratio=cfg.tile_max_empty_ratio,
            tile_rebuild=cfg.tile_rebuild,
            aug_enable=cfg.aug_enable,
            aug_policy=cfg.aug_policy,
            aug_train_copies=cfg.aug_train_copies,
            aug_verify_count=cfg.aug_verify_count,
            aug_verify_dir=aug_verify_dir,
            seed=cfg.seed,
        )

    if cfg.aug_verify_only:
        print("[INFO] aug_verify_only=True -> 타일/증강 생성만 수행하고 학습을 종료합니다.")
        if tile_meta is not None:
            print(json.dumps(tile_meta, ensure_ascii=False, indent=2))
        return

    train_aug_config, train_aug_source = resolve_train_aug_config(
        train_aug_preset=cfg.train_aug_preset,
        train_aug_config_json=cfg.train_aug_config_json,
    )

    train_ann, val_ann = assert_coco_layout(train_data_root)
    if cfg.sanity_check:
        quick_segmentation_sanity_check(train_ann)

    dataset_tag = train_data_root.name
    model_tag = f"Seg{cfg.size}_{cfg.resolution}"
    if cfg.tile_enable:
        model_tag += f"__TILE{cfg.tile_size}_ov{int(round(cfg.tile_overlap * 100)):02d}"
    if cfg.aug_enable:
        model_tag += f"__AUG_{cfg.aug_policy}_x{cfg.aug_train_copies}"
    if train_aug_source.startswith("custom:"):
        custom_stem = cfg.train_aug_config_json.stem if cfg.train_aug_config_json is not None else "custom"
        model_tag += f"__TRAUG_custom_{custom_stem}"
    elif train_aug_source != "default":
        model_tag += f"__TRAUG_{train_aug_source}"
    if cfg.gradient_checkpointing:
        model_tag += "__GC"
    if cfg.amp:
        model_tag += "__AMP"
    prefix = f"{dataset_tag}__{model_tag}"

    outputs_base = cfg.outputs_root / dataset_tag
    output_dir = create_output_dir(outputs_base, prefix=prefix)

    run_name = cfg.wandb_run_name or f"{model_tag}__{output_dir.name}"
    eff_batch = cfg.batch_size * cfg.grad_accum_steps

    print("====================================================")
    print("[INFO] RF-DETR Seg Training Start (trainV10)")
    print(f"[INFO] raw_data_root      : {cfg.data_root}")
    print(f"[INFO] train_data_root    : {train_data_root}")
    print(f"[INFO] train_ann          : {train_ann}")
    print(f"[INFO] val_ann            : {val_ann}")
    print(f"[INFO] output_dir         : {output_dir}")
    print(f"[INFO] dataset_tag        : {dataset_tag}")
    print(f"[INFO] model_tag          : {model_tag}")
    print(f"[INFO] size               : {cfg.size}")
    print(
        f"[INFO] resolution         : {cfg.resolution} (size mapping default)"
        if cfg.resolution == SIZE_TO_RESOLUTION[cfg.size]
        else f"[INFO] resolution         : {cfg.resolution} (override)"
    )
    print(f"[INFO] tile_enable        : {cfg.tile_enable}")
    if cfg.tile_enable:
        print(f"[INFO] tile_size          : {cfg.tile_size}")
        print(f"[INFO] tile_overlap       : {cfg.tile_overlap}")
        print(f"[INFO] tile_output_root   : {cfg.tile_output_root}")
    print(f"[INFO] tile_aug_enable    : {cfg.aug_enable}")
    if cfg.aug_enable:
        print(f"[INFO] tile_aug_policy    : {cfg.aug_policy}")
        print(f"[INFO] tile_aug_copies    : {cfg.aug_train_copies}")
        print(f"[INFO] tile_aug_verify_n  : {cfg.aug_verify_count}")
        print(f"[INFO] tile_aug_verify_dir: {aug_verify_dir}")
    print(f"[INFO] tile_aug_verify_only: {cfg.aug_verify_only}")
    print(f"[INFO] train_aug_preset   : {cfg.train_aug_preset}")
    print(f"[INFO] train_aug_source   : {train_aug_source}")
    if cfg.train_aug_config_json is not None:
        print(f"[INFO] train_aug_json     : {cfg.train_aug_config_json}")
    if train_aug_config is None:
        print("[INFO] train_aug_config   : default(AUG_CONFIG)")
    elif len(train_aug_config) == 0:
        print("[INFO] train_aug_config   : {} (disabled)")
    else:
        print(f"[INFO] train_aug_config   : {list(train_aug_config.keys())}")
    print(f"[INFO] epochs             : {cfg.epochs}")
    print(f"[INFO] batch_size         : {cfg.batch_size}")
    print(f"[INFO] grad_accum_steps   : {cfg.grad_accum_steps} (effective={eff_batch})")
    print(f"[INFO] num_workers        : {cfg.num_workers}")
    print(f"[INFO] pad_to_square      : {cfg.pad_to_square}")
    print(f"[INFO] lr                 : {cfg.lr}")
    print(f"[INFO] amp                : {cfg.amp}")
    print(f"[INFO] gradient_ckpt      : {cfg.gradient_checkpointing}")
    print(f"[INFO] patch_size         : {patch_size}")
    print(f"[INFO] num_windows        : {num_windows}")
    print(f"[INFO] pos_enc_size       : {positional_encoding_size}")
    print(f"[INFO] early_stopping     : {cfg.early_stopping}")
    print(f"[INFO] num_classes        : {cfg.num_classes}")
    print(f"[INFO] class_names        : {cfg.class_names}")
    print(f"[INFO] seed               : {cfg.seed}")
    if cfg.resume:
        print(f"[INFO] resume             : {cfg.resume}")
    print(f"[INFO] wandb              : {'OFF' if cfg.wandb_disable else 'ON (internal)'}")
    if not cfg.wandb_disable:
        print(f"[INFO] wandb_project      : {cfg.wandb_project}")
        print(f"[INFO] wandb_run          : {run_name}")
    print("====================================================")

    if torch.cuda.is_available():
        print(f"[INFO] CUDA device count  : {torch.cuda.device_count()}")
        print(f"[INFO] Current device     : {torch.cuda.current_device()}")
        print(f"[INFO] Device name        : {torch.cuda.get_device_name()}")
    else:
        print("[WARNING] CUDA is not available. Training will run on CPU.")

    run_meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "raw_data_root": str(cfg.data_root),
        "train_data_root": str(train_data_root),
        "train_ann": str(train_ann),
        "val_ann": str(val_ann),
        "output_dir": str(output_dir),
        "dataset_tag": dataset_tag,
        "model_size": cfg.size,
        "resolution": cfg.resolution,
        "resolution_source": "default" if cfg.resolution == SIZE_TO_RESOLUTION[cfg.size] else "override",
        "patch_size": patch_size,
        "num_windows": num_windows,
        "positional_encoding_size": positional_encoding_size,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "num_workers": cfg.num_workers,
        "pad_to_square": cfg.pad_to_square,
        "effective_batch": eff_batch,
        "lr": cfg.lr,
        "amp": cfg.amp,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "early_stopping": cfg.early_stopping,
        "num_classes": cfg.num_classes,
        "class_names": cfg.class_names,
        "seed": cfg.seed,
        "resume": cfg.resume,
        "tile_enable": cfg.tile_enable,
        "tile_size": cfg.tile_size if cfg.tile_enable else None,
        "tile_overlap": cfg.tile_overlap if cfg.tile_enable else None,
        "tile_min_area": cfg.tile_min_area if cfg.tile_enable else None,
        "tile_min_visible_ratio": cfg.tile_min_visible_ratio if cfg.tile_enable else None,
        "tile_max_empty_ratio": cfg.tile_max_empty_ratio if cfg.tile_enable else None,
        "tile_output_root": str(cfg.tile_output_root) if cfg.tile_enable else None,
        "tile_rebuild": cfg.tile_rebuild if cfg.tile_enable else None,
        "tile_meta": tile_meta,
        "tile_aug_enable": cfg.aug_enable,
        "tile_aug_policy": cfg.aug_policy if cfg.aug_enable else None,
        "tile_aug_train_copies": cfg.aug_train_copies if cfg.aug_enable else 0,
        "tile_aug_verify_count": cfg.aug_verify_count if cfg.aug_enable else 0,
        "tile_aug_verify_dir": str(aug_verify_dir) if (cfg.aug_enable and aug_verify_dir is not None) else None,
        "tile_aug_verify_only": cfg.aug_verify_only,
        "aug_enable": cfg.aug_enable,
        "aug_policy": cfg.aug_policy if cfg.aug_enable else None,
        "aug_train_copies": cfg.aug_train_copies if cfg.aug_enable else 0,
        "aug_verify_count": cfg.aug_verify_count if cfg.aug_enable else 0,
        "aug_verify_dir": str(aug_verify_dir) if (cfg.aug_enable and aug_verify_dir is not None) else None,
        "aug_verify_only": cfg.aug_verify_only,
        "train_aug_preset": cfg.train_aug_preset,
        "train_aug_source": train_aug_source,
        "train_aug_config_json": str(cfg.train_aug_config_json) if cfg.train_aug_config_json is not None else None,
        "train_aug_config": train_aug_config,
        "wandb_disable": cfg.wandb_disable,
        "wandb_project": cfg.wandb_project,
        "wandb_run": run_name,
        "argv": sys.argv,
        "python": sys.version,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    write_run_meta(output_dir, run_meta)

    use_wandb = True
    if not cfg.wandb_disable:
        try:
            import wandb as _wandb

            use_wandb = callable(getattr(_wandb, "init", None))
        except Exception:
            pass
        if not use_wandb:
            print("[WARN] wandb 사용 불가(미설치 또는 init 없음). wandb를 끕니다.")

    model = build_model(
        cfg.size,
        cfg.num_classes,
        cfg.class_names,
        resolution=cfg.resolution,
        amp=cfg.amp,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    train_kwargs = dict(
        dataset_dir=str(train_data_root),
        output_dir=str(output_dir),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        grad_accum_steps=cfg.grad_accum_steps,
        num_workers=cfg.num_workers,
        pad_to_square=cfg.pad_to_square,
        lr=cfg.lr,
        resolution=cfg.resolution,
        amp=cfg.amp,
        gradient_checkpointing=cfg.gradient_checkpointing,
        early_stopping=cfg.early_stopping,
        resume=cfg.resume,
        wandb=use_wandb,
        project=cfg.wandb_project if use_wandb else None,
        run=run_name if use_wandb else None,
    )
    if train_aug_config is not None:
        train_kwargs["aug_config"] = train_aug_config

    try:
        call_train(model, **train_kwargs)
    except RuntimeError as e:
        if "DataLoader worker" in str(e) and train_kwargs["num_workers"] > 0:
            print(
                "[WARN] DataLoader worker crashed. "
                f"Retrying once with num_workers=0 (was {train_kwargs['num_workers']})."
            )
            train_kwargs["num_workers"] = 0
            call_train(model, **train_kwargs)
        else:
            raise
    finally:
        cleanup_gpu_memory(model, verbose=True)

    print("====================================================")
    print("[INFO] Training finished.")
    print(f"[INFO] Output: {output_dir}")
    print("[INFO] Expected checkpoints:")
    print(f"  - {output_dir}/checkpoint_best_total.pth")
    print(f"  - {output_dir}/checkpoint_best_ema.pth")
    print(f"[INFO] Meta: {output_dir}/run_meta.json")
    print("====================================================")


if __name__ == "__main__":
    main()
