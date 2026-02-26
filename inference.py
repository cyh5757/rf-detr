#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF-DETR + SAHI sliced inference pipeline for cell dataset.

What this script does:
1) Load a fine-tuned RF-DETR segmentation checkpoint.
2) Pad each image to square with black background (no aspect-ratio distortion).
3) Run SAHI sliced inference on padded images.
4) Map predictions back to the original image space.
5) Save (under a run subfolder named from model/checkpoint by default):
   - padded image            -> padding/<run_name>/
   - visualization overlay   -> inferenced/<run_name>/
   - COCO annotation json    -> annote/<run_name>/_annotations.coco.json
   - summary json            -> annote/<run_name>/inference_summary.json

Overlay options (toggle):
- bbox / id / score / contour / mask fill
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Ensure local repository src has priority over any globally installed rfdetr package.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfdetr import (  # noqa: E402
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegXLarge,
)
from sahi.models.base import DetectionModel  # noqa: E402
from sahi.predict import get_sliced_prediction  # noqa: E402
from sahi.prediction import ObjectPrediction  # noqa: E402
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list  # noqa: E402
from sahi.utils.cv import get_coco_segmentation_from_bool_mask  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SIZE_TO_MODEL_CLASS = {
    "N": RFDETRSegNano,
    "S": RFDETRSegSmall,
    "M": RFDETRSegMedium,
    "L": RFDETRSegLarge,
    "XL": RFDETRSegXLarge,
    "2XL": RFDETRSeg2XLarge,
}


@dataclass
class InstanceRecord:
    category_id: int
    category_name: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    area: float
    segmentation: list[list[float]]
    mask: np.ndarray | None


def slugify(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\-\.]+", "_", text)  # spaces/specials -> _
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def default_run_name(ckpt_path: Path, size: str) -> str:
    parent = ckpt_path.parent.name
    base = parent if parent else ckpt_path.stem
    return slugify(f"{size}__{base}")


def color_for_index(index: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed=12345 + index * 9973)
    rgb = rng.integers(64, 256, size=3, dtype=np.int32)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def normalize_device_for_rfdetr(device: str) -> str:
    d = device.lower()
    if d.startswith("cuda"):
        return "cuda"
    if d.startswith("mps"):
        return "mps"
    return "cpu"


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).strip().lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def infer_category_mapping_from_checkpoint(model: Any) -> dict[str, str]:
    """
    Build category mapping from checkpoint args/class names.

    Handles common case where class_names = ["background", "cell"]
    and model predicts class_id=1 for "cell".
    """
    mapping: dict[str, str] = {}
    class_names = None

    try:
        args = getattr(getattr(model, "model", None), "args", None)
        if args is not None:
            class_names = getattr(args, "class_names", None)
    except Exception:
        class_names = None

    if class_names:
        names = [str(x) for x in class_names]
        if names and names[0].strip().lower() in {"background", "__background__", "bg"}:
            for cid in range(1, len(names)):
                mapping[str(cid)] = names[cid]
        else:
            for i, name in enumerate(names, start=1):
                mapping[str(i)] = name

    if not mapping:
        mapping = {"1": "cell"}

    return mapping


class RFDETRSahiDetectionModel(DetectionModel):
    """
    Minimal SAHI detection model wrapper for RF-DETR segmentation.
    """

    def __init__(self, model_size: str = "2XL", **kwargs):
        self.model_size = model_size.upper()
        self._original_shape: tuple[int, int] | None = None
        super().__init__(**kwargs)

    def load_model(self):
        if self.model_size not in SIZE_TO_MODEL_CLASS:
            raise ValueError(
                f"Unsupported model size: {self.model_size}. Use one of {list(SIZE_TO_MODEL_CLASS.keys())}."
            )
        model_cls = SIZE_TO_MODEL_CLASS[self.model_size]
        rfdetr_device = normalize_device_for_rfdetr(str(self.device))
        self.model = model_cls(
            pretrain_weights=self.model_path,
            device=rfdetr_device,
        )
        if self.category_mapping is None:
            self.category_mapping = infer_category_mapping_from_checkpoint(self.model)

    def set_model(self, model: Any, **kwargs):
        self.model = model
        if self.category_mapping is None:
            self.category_mapping = infer_category_mapping_from_checkpoint(self.model)

    def perform_inference(self, image: np.ndarray):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        inference_image = np.array(image, copy=True)
        self._original_shape = inference_image.shape[:2]
        self._original_predictions = self.model.predict(inference_image, threshold=self.confidence_threshold)

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ):
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        detections = self._original_predictions
        object_prediction_list_per_image: list[list[ObjectPrediction]] = []

        for image_ind, shift_amount in enumerate(shift_amount_list):
            full_shape = None
            if full_shape_list is not None:
                full_shape = full_shape_list[image_ind]
            elif self._original_shape is not None:
                full_shape = [self._original_shape[0], self._original_shape[1]]

            per_image_preds: list[ObjectPrediction] = []
            if detections is None or len(detections) == 0:
                object_prediction_list_per_image.append(per_image_preds)
                continue

            masks = getattr(detections, "mask", None)
            for idx in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[idx].tolist()
                score = float(detections.confidence[idx])
                category_id = int(detections.class_id[idx]) if detections.class_id is not None else 1
                category_name = self.category_mapping.get(str(category_id), f"class_{category_id}")

                if full_shape is not None:
                    x1 = float(np.clip(x1, 0, full_shape[1]))
                    x2 = float(np.clip(x2, 0, full_shape[1]))
                    y1 = float(np.clip(y1, 0, full_shape[0]))
                    y2 = float(np.clip(y2, 0, full_shape[0]))
                if x2 <= x1 or y2 <= y1:
                    continue

                segmentation = None
                if masks is not None:
                    bool_mask = masks[idx] > self.mask_threshold
                    segmentation = get_coco_segmentation_from_bool_mask(bool_mask)
                    if len(segmentation) == 0:
                        continue

                pred = ObjectPrediction(
                    bbox=[x1, y1, x2, y2],
                    category_id=category_id,
                    category_name=category_name,
                    score=score,
                    segmentation=segmentation,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                per_image_preds.append(pred)

            object_prediction_list_per_image.append(per_image_preds)

        self._object_prediction_list_per_image = object_prediction_list_per_image


def pad_to_square(image_rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = image_rgb.shape[:2]
    side = max(h, w)
    pad_top = (side - h) // 2
    pad_left = (side - w) // 2

    padded = np.zeros((side, side, 3), dtype=image_rgb.dtype)
    padded[pad_top : pad_top + h, pad_left : pad_left + w] = image_rgb
    return padded, pad_left, pad_top


def object_prediction_to_instance(
    obj_pred: ObjectPrediction,
    pad_left: int,
    pad_top: int,
    orig_h: int,
    orig_w: int,
    min_area: float,
) -> InstanceRecord | None:
    category_id = int(obj_pred.category.id)
    category_name = str(obj_pred.category.name)
    score = float(obj_pred.score.value)

    if obj_pred.mask is not None:
        full_mask = obj_pred.mask.bool_mask
        cropped_mask = full_mask[pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]
        area = float(cropped_mask.sum())
        if area < min_area:
            return None

        segmentation = get_coco_segmentation_from_bool_mask(cropped_mask)
        if len(segmentation) == 0:
            return None

        ys, xs = np.where(cropped_mask)
        if ys.size == 0 or xs.size == 0:
            return None
        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max() + 1)
        y2 = float(ys.max() + 1)
        return InstanceRecord(
            category_id=category_id,
            category_name=category_name,
            score=score,
            bbox_xyxy=(x1, y1, x2, y2),
            area=area,
            segmentation=segmentation,
            mask=cropped_mask.astype(bool),
        )

    # This script is for segmentation annotation generation, so drop bbox-only predictions.
    return None


def draw_visualization(
    image_rgb: np.ndarray,
    instances: list[InstanceRecord],
    *,
    draw_bbox: bool = True,
    draw_id: bool = True,
    draw_score: bool = True,
    draw_contour: bool = False,
    fill_mask: bool = True,
    alpha: float = 0.38,
    contour_thickness: int = 1,
    bbox_thickness: int = 1,
) -> np.ndarray:
    """
    Returns BGR overlay image.
    """
    vis_rgb = image_rgb.copy()

    # 1) mask fill (blend in RGB)
    if fill_mask:
        a = float(np.clip(alpha, 0.0, 1.0))
        for idx, inst in enumerate(instances, start=1):
            if inst.mask is None:
                continue
            color = np.array(color_for_index(idx), dtype=np.uint8)
            m = inst.mask
            vis_rgb[m] = ((1.0 - a) * vis_rgb[m] + a * color).astype(np.uint8)

    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    # 2) contour / bbox / text in BGR
    for idx, inst in enumerate(instances, start=1):
        color_rgb = color_for_index(idx)
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

        if draw_contour and inst.mask is not None:
            mask_u8 = (inst.mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(
                    vis_bgr,
                    contours,
                    -1,
                    color_bgr,
                    thickness=int(max(1, contour_thickness)),
                    lineType=cv2.LINE_AA,
                )

        if draw_bbox:
            x1, y1, x2, y2 = inst.bbox_xyxy
            p1 = (max(0, int(round(x1))), max(0, int(round(y1))))
            p2 = (max(0, int(round(x2))), max(0, int(round(y2))))
            cv2.rectangle(
                vis_bgr,
                p1,
                p2,
                color_bgr,
                int(max(1, bbox_thickness)),
                lineType=cv2.LINE_AA,
            )

        if draw_id or draw_score:
            parts: list[str] = []
            if draw_id:
                parts.append(f"#{idx}")
            if draw_score:
                parts.append(f"{inst.score:.2f}")
            label = " ".join(parts)

            x1, y1, _, _ = inst.bbox_xyxy
            p1 = (max(0, int(round(x1))), max(0, int(round(y1))))
            text_origin = (p1[0], max(14, p1[1] - 4))

            cv2.putText(
                vis_bgr,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_bgr,
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                vis_bgr,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

    return vis_bgr


def build_coco_payload(
    *,
    ckpt_path: Path,
    image_dir: Path,
    args: argparse.Namespace,
    coco_images: list[dict[str, Any]],
    coco_annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "info": {
            "description": "RF-DETR + SAHI inference generated annotations",
            "date_created": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": str(ckpt_path),
            "image_dir": str(image_dir),
            "padding_mode": "center black square",
            "run_name": getattr(args, "run_name", None),
            "model_size": args.size,
            "device": args.device,
            "thresholds": {
                "confidence_threshold": args.confidence_threshold,
                "mask_threshold": args.mask_threshold,
                "min_area": args.min_area,
            },
            "optimization": {
                "optimize_inference": bool(args.optimize_inference),
                "optimize_compile": bool(args.optimize_compile),
                "optimize_batch_size": int(args.optimize_batch_size),
                "optimize_dtype": str(args.optimize_dtype),
            },
            "sahi": {
                "slice_size": args.slice_size,
                "overlap": args.overlap,
                "postprocess_type": args.postprocess_type,
                "postprocess_match_metric": args.postprocess_match_metric,
                "postprocess_match_threshold": args.postprocess_match_threshold,
                "postprocess_class_agnostic": bool(args.postprocess_class_agnostic),
                "perform_standard_pred": bool(args.perform_standard_pred),
            },
            "visualization": {
                "vis_bbox": bool(args.vis_bbox),
                "vis_id": bool(args.vis_id),
                "vis_score": bool(args.vis_score),
                "vis_contour": bool(args.vis_contour),
                "vis_fill_mask": bool(args.vis_fill_mask),
                "vis_alpha": float(args.vis_alpha),
                "vis_contour_thickness": int(args.vis_contour_thickness),
                "vis_bbox_thickness": int(args.vis_bbox_thickness),
            },
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }


def write_coco_json(
    *,
    coco_path: Path,
    ckpt_path: Path,
    image_dir: Path,
    args: argparse.Namespace,
    coco_images: list[dict[str, Any]],
    coco_annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
) -> None:
    coco = build_coco_payload(
        ckpt_path=ckpt_path,
        image_dir=image_dir,
        args=args,
        coco_images=coco_images,
        coco_annotations=coco_annotations,
        categories=categories,
    )
    coco_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")


def load_existing_coco_state(coco_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str]]:
    if not coco_path.exists():
        return [], [], {}

    try:
        raw = json.loads(coco_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] existing coco json read failed ({coco_path}): {e}")
        return [], [], {}

    images = raw.get("images", [])
    annotations = raw.get("annotations", [])
    categories = raw.get("categories", [])

    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        print(f"[WARN] existing coco json format invalid. ignore and rebuild: {coco_path}")
        return [], [], {}

    category_table: dict[int, str] = {}
    for c in categories:
        try:
            cid = int(c.get("id"))
            name = str(c.get("name"))
        except Exception:
            continue
        category_table[cid] = name

    return images, annotations, category_table


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RF-DETR + SAHI sliced inference with square black padding.")

    # Python 3.9+ provides BooleanOptionalAction
    BoolOpt = argparse.BooleanOptionalAction

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/mbd1234/rf-detr/outputs/total_data_refine_1024/2026_02_25_1042__total_data_refine_1024__SegXL_624/checkpoint_best_ema.pth",
        help="Path to RF-DETR checkpoint (.pth/.pt).",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/home/mbd1234/data/Optiresolve_result_total_20260223/Image",
        help="Input image directory.",
    )
    parser.add_argument(
        "--annote-dir",
        type=str,
        default="/home/mbd1234/data/Optiresolve_result_total_20260223/annote",
        help="Base output directory for COCO annotation JSON (run subfolder will be created).",
    )
    parser.add_argument(
        "--padding-dir",
        type=str,
        default="/home/mbd1234/data/Optiresolve_result_total_20260223/padding",
        help="Base output directory for padded square images (run subfolder will be created).",
    )
    parser.add_argument(
        "--inferenced-dir",
        type=str,
        default="/home/mbd1234/data/Optiresolve_result_total_20260223/inferenced",
        help="Base output directory for visualization images (run subfolder will be created).",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output subfolder name. Default: '<size>__<checkpoint_parent_or_stem>'.",
    )

    parser.add_argument("--size", type=str, default="2XL", choices=list(SIZE_TO_MODEL_CLASS.keys()), help="RF-DETR size.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference: cuda/cuda:0/cpu/mps")
    parser.add_argument("--confidence-threshold", type=float, default=0.20, help="Detection confidence threshold.")
    parser.add_argument("--mask-threshold", type=float, default=0.50, help="Mask binarization threshold.")
    parser.add_argument(
        "--optimize-inference",
        action=BoolOpt,
        default=True,
        help="Enable RF-DETR optimize_for_inference() before sliced prediction.",
    )
    parser.add_argument(
        "--optimize-compile",
        action=BoolOpt,
        default=True,
        help="Use torch.jit.trace compilation in optimize_for_inference().",
    )
    parser.add_argument(
        "--optimize-batch-size",
        type=int,
        default=1,
        help="Batch size used when compiling optimized inference model.",
    )
    parser.add_argument(
        "--optimize-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for optimize_for_inference(). float32 uses more VRAM; float16 is usually faster.",
    )

    parser.add_argument("--slice-size", type=int, default=768, help="SAHI slice width/height in pixels.")
    parser.add_argument("--overlap", type=float, default=0.20, help="SAHI overlap ratio for both width/height.")
    parser.add_argument(
        "--postprocess-type",
        type=str,
        default="GREEDYNMM",
        choices=["NMM", "GREEDYNMM", "NMS", "LSNMS"],
        help="SAHI postprocess type.",
    )
    parser.add_argument("--postprocess-match-metric", type=str, default="IOS", choices=["IOU", "IOS"])
    parser.add_argument("--postprocess-match-threshold", type=float, default=0.5)
    parser.add_argument("--postprocess-class-agnostic", action="store_true", help="Merge regardless of class id.")
    parser.add_argument("--perform-standard-pred", action="store_true", help="Run full-image prediction in addition to slices.")

    parser.add_argument("--min-area", type=float, default=4.0, help="Minimum kept instance area in pixels.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images.")

    parser.add_argument("--save-padded", action=BoolOpt, default=True, help="Save padded square images.")
    parser.add_argument("--save-visuals", action=BoolOpt, default=True, help="Save visualization overlay images.")
    parser.add_argument("--skip-existing", action=BoolOpt, default=True, help="Skip files already present in existing COCO JSON.")
    parser.add_argument("--verbose-sahi", type=int, default=0, choices=[0, 1, 2], help="SAHI verbosity level.")
    parser.add_argument(
        "--autosave-interval",
        type=int,
        default=50,
        help="Save COCO JSON every N newly processed images. Set 0 to disable interim autosave.",
    )

    # Visualization toggles
    parser.add_argument("--vis-bbox", action=BoolOpt, default=True, help="Draw bbox on overlay.")
    parser.add_argument("--vis-id", action=BoolOpt, default=True, help="Draw instance id on overlay.")
    parser.add_argument("--vis-score", action=BoolOpt, default=True, help="Draw score text on overlay.")
    parser.add_argument("--vis-contour", action=BoolOpt, default=False, help="Draw mask contour on overlay.")
    parser.add_argument("--vis-fill-mask", action=BoolOpt, default=True, help="Alpha-blend mask fill.")

    parser.add_argument("--vis-alpha", type=float, default=0.38, help="Mask overlay alpha (0~1).")
    parser.add_argument("--vis-contour-thickness", type=int, default=2, help="Contour thickness.")
    parser.add_argument("--vis-bbox-thickness", type=int, default=1, help="BBox thickness.")

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if str(args.device).lower().startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Run name (subfolder) based on model/checkpoint unless user overrides
    run_name = args.run_name or default_run_name(ckpt_path, args.size)
    args.run_name = run_name  # keep in args for COCO info

    # Base output dirs -> run subdirs
    annote_dir_base = Path(args.annote_dir).expanduser().resolve()
    padding_dir_base = Path(args.padding_dir).expanduser().resolve()
    inferenced_dir_base = Path(args.inferenced_dir).expanduser().resolve()

    annote_dir = annote_dir_base / run_name
    padding_dir = padding_dir_base / run_name
    inferenced_dir = inferenced_dir_base / run_name
    coco_path = annote_dir / "_annotations.coco.json"

    annote_dir.mkdir(parents=True, exist_ok=True)
    padding_dir.mkdir(parents=True, exist_ok=True)
    inferenced_dir.mkdir(parents=True, exist_ok=True)

    save_padded = bool(args.save_padded)
    save_visuals = bool(args.save_visuals)
    skip_existing = bool(args.skip_existing)

    image_paths = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    category_table: dict[int, str] = {}
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    processed_filenames: set[str] = set()
    next_image_id = 1
    ann_id = 1

    if skip_existing:
        existing_images, existing_annotations, existing_categories = load_existing_coco_state(coco_path)
        if existing_images or existing_annotations or existing_categories:
            coco_images.extend(existing_images)
            coco_annotations.extend(existing_annotations)
            processed_filenames = {
                str(img.get("file_name")) for img in coco_images if isinstance(img, dict) and img.get("file_name")
            }
            category_table.update(existing_categories)
            if coco_images:
                next_image_id = max(int(img.get("id", 0)) for img in coco_images if isinstance(img, dict)) + 1
            if coco_annotations:
                ann_id = max(int(ann.get("id", 0)) for ann in coco_annotations if isinstance(ann, dict)) + 1
            print(
                f"[INFO] resume existing coco: images={len(coco_images)}, "
                f"annotations={len(coco_annotations)}, next_image_id={next_image_id}, next_ann_id={ann_id}"
            )

    pending_image_paths = image_paths
    if skip_existing:
        pending_image_paths = [p for p in image_paths if p.name not in processed_filenames]

    print(f"[INFO] run_name        : {run_name}")
    print(f"[INFO] checkpoint      : {ckpt_path}")
    print(f"[INFO] image_dir       : {image_dir}")
    print(f"[INFO] total_images    : {len(image_paths)}")
    print(f"[INFO] pending_images  : {len(pending_image_paths)}")
    print(f"[INFO] padding mode    : black square center padding")
    print(f"[INFO] sahi slice      : {args.slice_size}x{args.slice_size}, overlap={args.overlap}")
    print(f"[INFO] skip_existing   : {skip_existing}")
    print(f"[INFO] out annote      : {annote_dir}")
    print(f"[INFO] out padding     : {padding_dir}")
    print(f"[INFO] out inferenced  : {inferenced_dir}")

    skipped_existing_count = 0
    new_processed_count = 0

    if skip_existing:
        skipped_existing_count = len(image_paths) - len(pending_image_paths)

    if len(pending_image_paths) == 0:
        if not category_table:
            category_table = {1: "cell"}
        categories = [{"id": cid, "name": name, "supercategory": "cell"} for cid, name in sorted(category_table.items())]
        write_coco_json(
            coco_path=coco_path,
            ckpt_path=ckpt_path,
            image_dir=image_dir,
            args=args,
            coco_images=coco_images,
            coco_annotations=coco_annotations,
            categories=categories,
        )
        summary = {
            "run_name": run_name,
            "images": len(coco_images),
            "annotations": len(coco_annotations),
            "categories": categories,
            "coco_json": str(coco_path),
            "padding_dir": str(padding_dir),
            "inferenced_dir": str(inferenced_dir),
            "save_padded": save_padded,
            "save_visuals": save_visuals,
            "skip_existing": skip_existing,
            "skipped_existing_images": skipped_existing_count,
            "newly_processed_images": 0,
        }
        summary_path = annote_dir / "inference_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[DONE] No pending images. All existing results were skipped.")
        print(f"[DONE] COCO json     : {coco_path}")
        print(f"[DONE] Summary json  : {summary_path}")
        print(f"[DONE] #images       : {len(coco_images)}")
        print(f"[DONE] #annotations  : {len(coco_annotations)}")
        return

    detection_model = RFDETRSahiDetectionModel(
        model_size=args.size,
        model_path=str(ckpt_path),
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        mask_threshold=args.mask_threshold,
        load_at_init=True,
    )

    if bool(args.optimize_inference):
        optimize_dtype = resolve_torch_dtype(args.optimize_dtype)
        optimize_batch_size = max(1, int(args.optimize_batch_size))
        try:
            detection_model.model.optimize_for_inference(
                compile=bool(args.optimize_compile),
                batch_size=optimize_batch_size,
                dtype=optimize_dtype,
            )
            print(
                f"[INFO] optimize_for_inference: enabled "
                f"(compile={bool(args.optimize_compile)}, batch_size={optimize_batch_size}, dtype={args.optimize_dtype})"
            )
        except Exception as e:
            print(f"[WARN] optimize_for_inference failed. Fallback to regular inference: {e}")
    else:
        print("[INFO] optimize_for_inference: disabled")

    if detection_model.category_mapping:
        for k, v in detection_model.category_mapping.items():
            try:
                cid = int(k)
            except Exception:
                continue
            category_table.setdefault(cid, str(v))

    for img_path in tqdm(pending_image_paths, desc="Inference"):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        padded_rgb, pad_left, pad_top = pad_to_square(rgb)
        if save_padded:
            padded_bgr = cv2.cvtColor(padded_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(padding_dir / img_path.name), padded_bgr)

        sliced_result = get_sliced_prediction(
            image=padded_rgb,
            detection_model=detection_model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            perform_standard_pred=args.perform_standard_pred,
            postprocess_type=args.postprocess_type,
            postprocess_match_metric=args.postprocess_match_metric,
            postprocess_match_threshold=args.postprocess_match_threshold,
            postprocess_class_agnostic=args.postprocess_class_agnostic,
            auto_slice_resolution=False,
            verbose=args.verbose_sahi,
        )

        instances: list[InstanceRecord] = []
        for obj_pred in sliced_result.object_prediction_list:
            inst = object_prediction_to_instance(
                obj_pred=obj_pred,
                pad_left=pad_left,
                pad_top=pad_top,
                orig_h=h,
                orig_w=w,
                min_area=args.min_area,
            )
            if inst is None:
                continue
            instances.append(inst)
            category_table.setdefault(inst.category_id, inst.category_name)

        image_id = next_image_id
        next_image_id += 1
        coco_images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            }
        )

        for inst in instances:
            x1, y1, x2, y2 = inst.bbox_xyxy
            bw = max(0.0, float(x2 - x1))
            bh = max(0.0, float(y2 - y1))
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(inst.category_id),
                    "bbox": [float(x1), float(y1), bw, bh],
                    "area": float(inst.area),
                    "segmentation": inst.segmentation,
                    "iscrowd": 0,
                    "score": float(inst.score),
                }
            )
            ann_id += 1

        if save_visuals:
            vis_bgr = draw_visualization(
                rgb,
                instances,
                draw_bbox=bool(args.vis_bbox),
                draw_id=bool(args.vis_id),
                draw_score=bool(args.vis_score),
                draw_contour=bool(args.vis_contour),
                fill_mask=bool(args.vis_fill_mask),
                alpha=float(args.vis_alpha),
                contour_thickness=int(args.vis_contour_thickness),
                bbox_thickness=int(args.vis_bbox_thickness),
            )
            cv2.imwrite(str(inferenced_dir / img_path.name), vis_bgr)

        processed_filenames.add(img_path.name)
        new_processed_count += 1

        if args.autosave_interval > 0 and (new_processed_count % args.autosave_interval == 0):
            categories_for_save = [
                {"id": cid, "name": name, "supercategory": "cell"} for cid, name in sorted(category_table.items())
            ]
            write_coco_json(
                coco_path=coco_path,
                ckpt_path=ckpt_path,
                image_dir=image_dir,
                args=args,
                coco_images=coco_images,
                coco_annotations=coco_annotations,
                categories=categories_for_save if categories_for_save else [{"id": 1, "name": "cell", "supercategory": "cell"}],
            )

    if not category_table:
        category_table = {1: "cell"}
    categories = [{"id": cid, "name": name, "supercategory": "cell"} for cid, name in sorted(category_table.items())]
    write_coco_json(
        coco_path=coco_path,
        ckpt_path=ckpt_path,
        image_dir=image_dir,
        args=args,
        coco_images=coco_images,
        coco_annotations=coco_annotations,
        categories=categories,
    )

    summary = {
        "run_name": run_name,
        "images": len(coco_images),
        "annotations": len(coco_annotations),
        "categories": categories,
        "coco_json": str(coco_path),
        "padding_dir": str(padding_dir),
        "inferenced_dir": str(inferenced_dir),
        "save_padded": save_padded,
        "save_visuals": save_visuals,
        "skip_existing": skip_existing,
        "skipped_existing_images": skipped_existing_count,
        "newly_processed_images": new_processed_count,
    }
    summary_path = annote_dir / "inference_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] Inference finished.")
    print(f"[DONE] COCO json     : {coco_path}")
    print(f"[DONE] Summary json  : {summary_path}")
    print(f"[DONE] #images       : {len(coco_images)}")
    print(f"[DONE] #annotations  : {len(coco_annotations)}")


if __name__ == "__main__":
    main()
