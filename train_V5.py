#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RF-DETR Segmentation Training Script (cell_opti_10) - v5

v4.3 -> v5 변경점:
- backbone freeze + query/decoder/head 중심 파인튜닝 모드 추가
- tune mode 선택 (--tune-mode: query_only/query_decoder_heads/full)
- 실제 trainable 파라미터 요약 로그 + run_meta.json 기록
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import weakref
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch


TUNE_MODE_QUERY_ONLY = "query_only"
TUNE_MODE_QUERY_DECODER_HEADS = "query_decoder_heads"
TUNE_MODE_FULL = "full"
TUNE_MODE_CHOICES = [
    TUNE_MODE_QUERY_ONLY,
    TUNE_MODE_QUERY_DECODER_HEADS,
    TUNE_MODE_FULL,
]

QUERY_PARAM_KEYWORDS = (
    "query_feat",
    "refpoint_embed",
)

DECODER_HEAD_PARAM_KEYWORDS = (
    "transformer.decoder",
    "class_embed",
    "bbox_embed",
    "segmentation_head",
    "transformer.enc_out_class_embed",
    "transformer.enc_out_bbox_embed",
)


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

SIZE_TO_DEFAULT_NUM_QUERIES = {
    "N": 100,
    "S": 100,
    "M": 200,
    "L": 200,
    "XL": 300,
    "2XL": 300,
}

SIZE_TO_DEFAULT_NUM_SELECT = {
    "N": 100,
    "S": 100,
    "M": 200,
    "L": 200,
    "XL": 300,
    "2XL": 300,
}

SIZE_TO_PRETRAIN_WEIGHT_NAME = {
    "N": "rf-detr-seg-nano.pt",
    "S": "rf-detr-seg-small.pt",
    "M": "rf-detr-seg-medium.pt",
    "L": "rf-detr-seg-large.pt",
    "XL": "rf-detr-seg-xlarge.pt",
    "2XL": "rf-detr-seg-xxlarge.pt",
}


# -----------------------------
# 2) output_dir 생성
# -----------------------------
def create_output_dir(base_dir: str | Path, prefix: str) -> Path:
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    out_dir = base_dir / f"{timestamp}__{prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# -----------------------------
# 3) GPU 메모리 정리
# -----------------------------
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


# -----------------------------
# 3.5) Partial pretrained load (shape-compatible only)
# -----------------------------
def load_compatible_weights_into_torch_module(
    module: torch.nn.Module,
    checkpoint_path: str,
    verbose: bool = True,
) -> Tuple[int, int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrained = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model_sd = module.state_dict()
    compatible = {k: v for k, v in pretrained.items() if k in model_sd and v.shape == model_sd[k].shape}

    module.load_state_dict(compatible, strict=False)

    total = len(model_sd)
    loaded = len(compatible)

    if verbose:
        print(f"[PARTIAL LOAD] checkpoint: {checkpoint_path}")
        print(f"[PARTIAL LOAD] Loaded {loaded}/{total} tensors (shape-compatible)")
        print(f"[PARTIAL LOAD] Skipped {total - loaded} tensors (mismatch/missing)")
        skipped_keys = [k for k in model_sd.keys() if k not in compatible]
        if skipped_keys:
            print(f"[PARTIAL LOAD] Example skipped keys (up to 12): {skipped_keys[:12]}")

    return loaded, total


def find_inner_torch_module(model) -> torch.nn.Module:
    candidates = [
        "model.model.model",
        "model.model",
        "model",
    ]

    for path in candidates:
        cur = model
        ok = True
        for a in path.split("."):
            if not hasattr(cur, a):
                ok = False
                break
            cur = getattr(cur, a)
        if ok and isinstance(cur, torch.nn.Module):
            print(f"[INFO] Found torch module at: {path}")
            return cur

    raise RuntimeError(
        "내부 torch.nn.Module 경로를 찾지 못했습니다. model 구조를 확인해야 합니다(버전 차이 가능)."
    )


# -----------------------------
# 3.6) rfdetr hosted-key 자동 다운로드 지원
# -----------------------------
def _is_none_like(x: Optional[str]) -> bool:
    return x is None or str(x).strip().lower() in {"", "none", "null"}


def ensure_pretrained_weights(maybe_hosted_key_or_path: Optional[str], redownload: bool = False) -> Optional[str]:
    if _is_none_like(maybe_hosted_key_or_path):
        return None

    s = str(maybe_hosted_key_or_path)

    if os.path.exists(s):
        return s

    try:
        from rfdetr.main import download_pretrain_weights
        download_pretrain_weights(s, redownload=redownload)
        if os.path.exists(s):
            print(f"[WEIGHTS] Downloaded: {s}")
        else:
            print(f"[WEIGHTS][WARN] download_pretrain_weights called but file not found: {s}")
        return s
    except Exception as e:
        print(f"[WEIGHTS][WARN] download_pretrain_weights import/call failed: {e}")
        return s


def _safe_torch_load_checkpoint(path: str):
    import argparse as _argparse
    try:
        torch.serialization.add_safe_globals([_argparse.Namespace])
    except Exception:
        pass

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def inspect_query_slots_from_checkpoint(checkpoint_path: str) -> dict:
    out = {
        "path": str(checkpoint_path),
        "exists": False,
        "ok": False,
        "total_query_slots": None,
        "refpoint_shape": None,
        "query_feat_shape": None,
        "error": None,
    }
    if not checkpoint_path:
        out["error"] = "empty path"
        return out

    p = Path(checkpoint_path)
    if (not p.exists()) and (not p.is_absolute()):
        alt = Path(__file__).resolve().parent / p
        if alt.exists():
            p = alt

    out["exists"] = p.exists()
    if not p.exists():
        out["error"] = "file not found"
        return out

    try:
        ckpt = _safe_torch_load_checkpoint(str(p))
        model_state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(model_state, dict):
            out["error"] = f"unexpected checkpoint type: {type(model_state)}"
            return out

        rp = model_state.get("refpoint_embed.weight", None)
        qf = model_state.get("query_feat.weight", None)
        if rp is None:
            out["error"] = "refpoint_embed.weight not found"
            return out

        out["refpoint_shape"] = tuple(rp.shape)
        out["query_feat_shape"] = tuple(qf.shape) if qf is not None else None
        out["total_query_slots"] = int(rp.shape[0])
        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = repr(e)
        return out


def print_query_capacity_report(paths: list[str], group_detr: int) -> None:
    print("====================================================")
    print("[QUERY CAPACITY REPORT]")
    print(f"[INFO] group_detr used for conversion: {group_detr}")
    print("----------------------------------------------------")
    for p in paths:
        info = inspect_query_slots_from_checkpoint(p)
        print(f"- ckpt: {p}")
        if not info["ok"]:
            print(f"  status: FAIL ({info['error']})")
            continue

        total_slots = int(info["total_query_slots"])
        if group_detr > 0 and total_slots % group_detr == 0:
            max_nq = total_slots // group_detr
            cap_msg = f"max_num_queries={max_nq} (total_slots={total_slots}, group_detr={group_detr})"
        else:
            cap_msg = f"max_num_queries=unknown (total_slots={total_slots} not divisible by group_detr={group_detr})"

        print("  status: OK")
        print(f"  refpoint_embed.weight: {info['refpoint_shape']}")
        print(f"  query_feat.weight   : {info['query_feat_shape']}")
        print(f"  inferred capacity   : {cap_msg}")
    print("====================================================")


# -----------------------------
# 4) 데이터 검증
# -----------------------------
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
            "샘플에서 유효한 segmentation polygon을 하나도 못 찾음. COCO segmentation 필드/라벨링 상태 확인 필요."
        )

    print(f"[SANITY] segmentation polygons found: {ok}/{checked} (sample)")


# -----------------------------
# 5) 모델 생성
# -----------------------------
def build_model(
    size: str,
    num_classes: int,
    class_names: list[str],
    num_queries: Optional[int] = None,
    num_select: Optional[int] = None,
    group_detr: Optional[int] = None,
    pretrain_weights: Optional[str] = None,
    disable_pretrain_weights: bool = False,
):
    import rfdetr

    size = size.upper()

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
            }
            if num_queries is not None:
                kwargs["num_queries"] = int(num_queries)
            if num_select is not None:
                kwargs["num_select"] = int(num_select)
            if group_detr is not None:
                kwargs["group_detr"] = int(group_detr)

            if disable_pretrain_weights:
                kwargs["pretrain_weights"] = None
            elif pretrain_weights is not None:
                kwargs["pretrain_weights"] = str(pretrain_weights)

            return cls(**kwargs)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"모델 생성 실패. candidates={candidates}, last_err={last_err}")


# -----------------------------
# 6) train() 호출: SegmentationTrainConfig(model_fields) 키만 전달
# -----------------------------
def call_train(model, verbose_filter: bool = True, **kwargs):
    from rfdetr.config import SegmentationTrainConfig

    allowed = set(SegmentationTrainConfig.model_fields.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    dropped = {k: v for k, v in kwargs.items() if k not in allowed and v is not None}

    if verbose_filter:
        print("[TRAIN ARGS] Allowed keys count:", len(allowed))
        print("[TRAIN ARGS] Passed keys:", sorted(filtered.keys()))
        if dropped:
            # 일부러 보여줌: 지금까지 "resolution 태그만"이거나 옵션이 버려진 경우를 잡기 위함
            print("[TRAIN ARGS][WARN] Dropped keys (not in SegmentationTrainConfig):", sorted(dropped.keys()))

    return model.train(**filtered)


# -----------------------------
# 6.5) Fine-tuning policy (freeze/unfreeze)
# -----------------------------
def _contains_any(name: str, keywords: tuple[str, ...]) -> bool:
    return any(k in name for k in keywords)


def _resolve_unfreeze_keywords(tune_mode: str) -> tuple[str, ...]:
    if tune_mode == TUNE_MODE_QUERY_ONLY:
        return QUERY_PARAM_KEYWORDS
    if tune_mode == TUNE_MODE_QUERY_DECODER_HEADS:
        return QUERY_PARAM_KEYWORDS + DECODER_HEAD_PARAM_KEYWORDS
    if tune_mode == TUNE_MODE_FULL:
        return tuple()
    raise ValueError(f"Unsupported tune_mode: {tune_mode}")


def apply_finetune_policy(
    module: torch.nn.Module,
    tune_mode: str,
    freeze_backbone: bool = True,
):
    total_params = 0
    trainable_params = 0
    total_tensors = 0
    trainable_tensors = 0
    trainable_names: list[str] = []
    frozen_names: list[str] = []

    unfreeze_keywords = _resolve_unfreeze_keywords(tune_mode)

    for name, p in module.named_parameters():
        total_tensors += 1
        total_params += p.numel()

        if tune_mode == TUNE_MODE_FULL:
            should_train = True
        else:
            should_train = _contains_any(name, unfreeze_keywords)

        if freeze_backbone and "backbone" in name:
            should_train = False

        p.requires_grad = bool(should_train)

        if p.requires_grad:
            trainable_tensors += 1
            trainable_params += p.numel()
            trainable_names.append(name)
        else:
            frozen_names.append(name)

    return {
        "tune_mode": tune_mode,
        "freeze_backbone": bool(freeze_backbone),
        "total_tensors": total_tensors,
        "trainable_tensors": trainable_tensors,
        "frozen_tensors": total_tensors - trainable_tensors,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "trainable_ratio": (trainable_params / total_params) if total_params > 0 else 0.0,
        "trainable_name_samples": trainable_names[:30],
        "frozen_name_samples": frozen_names[:30],
    }


def print_finetune_policy_summary(summary: dict) -> None:
    print("----------------------------------------------------")
    print("[FINETUNE POLICY]")
    print(f"[INFO] tune_mode          : {summary['tune_mode']}")
    print(f"[INFO] freeze_backbone    : {summary['freeze_backbone']}")
    print(
        f"[INFO] trainable tensors  : {summary['trainable_tensors']}/{summary['total_tensors']} "
        f"(frozen={summary['frozen_tensors']})"
    )
    print(
        f"[INFO] trainable params   : {summary['trainable_params']}/{summary['total_params']} "
        f"({summary['trainable_ratio'] * 100:.2f}%)"
    )
    print(f"[INFO] trainable samples  : {summary['trainable_name_samples'][:12]}")
    print(f"[INFO] frozen samples     : {summary['frozen_name_samples'][:12]}")
    print("----------------------------------------------------")


# -----------------------------
# 7) seed 고정
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# 8) run meta 저장
# -----------------------------
def write_run_meta(output_dir: Path, meta: dict) -> None:
    p = output_dir / "run_meta.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# 9) Args
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
    lr: float
    device: str
    amp: bool
    gradient_checkpointing: bool
    group_detr: int

    num_classes: int
    class_names: list[str]
    early_stopping: bool
    resume: Optional[str]
    seed: int
    sanity_check: bool

    wandb_project: Optional[str]
    wandb_run_name: Optional[str]
    wandb_disable: bool

    num_queries: Optional[int]
    num_select: Optional[int]
    tune_mode: str
    freeze_backbone: bool
    partial_load: bool
    pretrained_ckpt: Optional[str]
    pretrain_weights: Optional[str]
    inspect_ckpts: Optional[list[str]]
    dry_run: bool

    alloc_expandable_segments: bool


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="RF-DETR Segmentation training (cell_opti_10) - v5")

    p.add_argument("--data-root", type=str, default="/home/mbd1234/data/Optiresolve_result_20260212/cell_opti_10")
    p.add_argument("--outputs-root", type=str, default="/home/mbd1234/rf-detr/outputs")

    p.add_argument("--size", type=str, default="M", choices=list(SIZE_TO_RESOLUTION.keys()))
    p.add_argument("--resolution", type=int, default=None,
                   help="입력 해상도. 미지정 시 size 기본값. (train()로 전달)")

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers. Set 0/1 if worker gets killed.")
    p.add_argument("--lr", type=float, default=5e-5)

    # ✅ 메모리/성능 옵션
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--amp", action="store_true", help="혼합정밀(fp16/bf16) 사용(가능하면 메모리 크게 절감)")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Recompute로 activation 저장 줄여 메모리 절감 (느려짐)")
    p.add_argument("--group-detr", type=int, default=13,
                   help="group DETR groups (training-time grouped queries).")

    # 파편화 완화 환경변수 자동 설정
    p.add_argument("--alloc-expandable-segments", action="store_true",
                   help="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 설정(파편화 완화)")

    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--class-names", type=str, nargs="+", default=["background", "cell"])

    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sanity-check", action="store_true")

    # wandb
    p.add_argument("--wandb-project", type=str, default="cell_rfdetr")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-disable", action="store_true")

    # num_queries / num_select / partial-load
    p.add_argument("--num-queries", type=int, default=None, help="DETR queries 수(상한 증가 목적)")
    p.add_argument("--num-select", type=int, default=None, help="PostProcess top-k (최종 출력 상한)")
    p.add_argument(
        "--tune-mode",
        type=str,
        default=TUNE_MODE_QUERY_DECODER_HEADS,
        choices=TUNE_MODE_CHOICES,
        help=(
            "query_only: query 임베딩만 학습 / "
            "query_decoder_heads: query+decoder+head 학습(기본) / "
            "full: 전체 학습"
        ),
    )
    p.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true",
                   help="backbone 파라미터를 강제로 freeze")
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                   help="backbone freeze를 비활성화")
    p.set_defaults(freeze_backbone=True)
    p.add_argument("--partial-load", action="store_true", help="shape 동일한 pretrained weight만 부분 로딩")
    p.add_argument("--pretrained-ckpt", type=str, default=None, help="부분 로딩할 pretrained ckpt 경로 또는 hosted key")

    # wrapper pretrain_weights override (hosted key 지원)
    p.add_argument("--pretrain-weights", type=str, default=None,
                   help="wrapper pretrain_weights override (hosted key/path/'none')")
    p.add_argument("--inspect-ckpts", type=str, nargs="+", default=None,
                   help="checkpoint(s) to inspect query-slot capacity and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="print resolved training plan and exit without training")

    a = p.parse_args()

    size = a.size.upper()
    resolution = a.resolution if a.resolution is not None else SIZE_TO_RESOLUTION[size]

    return TrainConfig(
        data_root=Path(a.data_root),
        outputs_root=Path(a.outputs_root),
        size=size,
        resolution=resolution,
        epochs=a.epochs,
        batch_size=a.batch_size,
        grad_accum_steps=a.grad_accum_steps,
        num_workers=int(a.num_workers),
        lr=a.lr,
        device=a.device,
        amp=bool(a.amp),
        gradient_checkpointing=bool(a.gradient_checkpointing),
        group_detr=int(a.group_detr),
        num_classes=a.num_classes,
        class_names=a.class_names,
        early_stopping=bool(a.early_stopping),
        resume=a.resume,
        seed=a.seed,
        sanity_check=bool(a.sanity_check),
        wandb_project=None if a.wandb_disable else a.wandb_project,
        wandb_run_name=a.wandb_run_name,
        wandb_disable=bool(a.wandb_disable),
        num_queries=a.num_queries,
        num_select=a.num_select,
        tune_mode=str(a.tune_mode),
        freeze_backbone=bool(a.freeze_backbone),
        partial_load=bool(a.partial_load),
        pretrained_ckpt=a.pretrained_ckpt,
        pretrain_weights=a.pretrain_weights,
        inspect_ckpts=a.inspect_ckpts,
        dry_run=bool(a.dry_run),
        alloc_expandable_segments=bool(a.alloc_expandable_segments),
    )


# -----------------------------
# 10) Main
# -----------------------------
def main():
    cfg = parse_args()

    if cfg.inspect_ckpts:
        print_query_capacity_report(cfg.inspect_ckpts, group_detr=int(cfg.group_detr))
        return

    # 파편화 완화 환경변수(원할 때만)
    if cfg.alloc_expandable_segments:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print("[ENV] PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    set_seed(cfg.seed)

    train_ann, val_ann = assert_coco_layout(cfg.data_root)
    if cfg.sanity_check:
        quick_segmentation_sanity_check(train_ann)

    if cfg.group_detr <= 0:
        raise ValueError(f"--group-detr must be > 0, got {cfg.group_detr}")

    default_nq = int(SIZE_TO_DEFAULT_NUM_QUERIES[cfg.size])
    default_ns = int(SIZE_TO_DEFAULT_NUM_SELECT[cfg.size])
    effective_num_queries = int(cfg.num_queries) if cfg.num_queries is not None else default_nq
    if cfg.num_select is not None:
        effective_num_select = int(cfg.num_select)
    else:
        # If num_queries is overridden, keep postprocess cap aligned unless user explicitly sets num_select.
        effective_num_select = effective_num_queries if cfg.num_queries is not None else default_ns

    if effective_num_queries <= 0:
        raise ValueError(f"num_queries must be > 0, got {effective_num_queries}")
    if effective_num_select <= 0:
        raise ValueError(f"num_select must be > 0, got {effective_num_select}")
    if effective_num_select > effective_num_queries:
        raise ValueError(
            f"num_select({effective_num_select}) cannot exceed num_queries({effective_num_queries})."
        )
    dataset_tag = cfg.data_root.name
    model_tag = f"Seg{cfg.size}_{cfg.resolution}"
    model_tag = f"{model_tag}__Q{effective_num_queries}__S{effective_num_select}__G{cfg.group_detr}"
    tune_mode_short = {
        TUNE_MODE_QUERY_ONLY: "QOnly",
        TUNE_MODE_QUERY_DECODER_HEADS: "QDecHead",
        TUNE_MODE_FULL: "Full",
    }[cfg.tune_mode]
    model_tag = f"{model_tag}__FT{tune_mode_short}"
    if cfg.freeze_backbone:
        model_tag = f"{model_tag}__BBFrozen"
    if cfg.gradient_checkpointing:
        model_tag = f"{model_tag}__GCkpt"
    if cfg.amp:
        model_tag = f"{model_tag}__AMP"

    prefix = f"{dataset_tag}__{model_tag}"

    outputs_base = cfg.outputs_root / dataset_tag
    output_dir = create_output_dir(outputs_base, prefix=prefix)

    run_name = cfg.wandb_run_name or f"{model_tag}__{output_dir.name}"
    eff_batch = cfg.batch_size * cfg.grad_accum_steps

    print("====================================================")
    print("[INFO] RF-DETR Seg Training Start (v5)")
    print(f"[INFO] data_root               : {cfg.data_root}")
    print(f"[INFO] train_ann               : {train_ann}")
    print(f"[INFO] val_ann                 : {val_ann}")
    print(f"[INFO] output_dir              : {output_dir}")
    print(f"[INFO] dataset_tag             : {dataset_tag}")
    print(f"[INFO] model_tag               : {model_tag}")
    print(f"[INFO] size                    : {cfg.size}")
    print(f"[INFO] resolution(train input) : {cfg.resolution}")
    print(f"[INFO] num_queries (requested) : {cfg.num_queries}")
    print(f"[INFO] num_queries (effective) : {effective_num_queries} (default={default_nq})")
    print(f"[INFO] num_select  (requested) : {cfg.num_select}")
    print(f"[INFO] num_select  (effective) : {effective_num_select} (default={default_ns})")
    print(f"[INFO] group_detr              : {cfg.group_detr}")
    print(f"[INFO] tune_mode               : {cfg.tune_mode}")
    print(f"[INFO] freeze_backbone         : {cfg.freeze_backbone}")
    print(f"[INFO] epochs                  : {cfg.epochs}")
    print(f"[INFO] batch_size              : {cfg.batch_size}")
    print(f"[INFO] grad_accum_steps        : {cfg.grad_accum_steps} (effective={eff_batch})")
    print(f"[INFO] num_workers             : {cfg.num_workers}")
    print(f"[INFO] lr                      : {cfg.lr}")
    print(f"[INFO] device                  : {cfg.device}")
    print(f"[INFO] amp                     : {cfg.amp}")
    print(f"[INFO] gradient_checkpointing  : {cfg.gradient_checkpointing}")
    print(f"[INFO] early_stopping          : {cfg.early_stopping}")
    print(f"[INFO] num_classes             : {cfg.num_classes}")
    print(f"[INFO] class_names             : {cfg.class_names}")
    print(f"[INFO] seed                    : {cfg.seed}")
    if cfg.resume:
        print(f"[INFO] resume                  : {cfg.resume}")
    print(f"[INFO] pretrain_weights arg    : {cfg.pretrain_weights}")
    print(f"[INFO] partial_load            : {cfg.partial_load}")
    if cfg.partial_load:
        print(f"[INFO] pretrained_ckpt         : {cfg.pretrained_ckpt}")
    print("====================================================")

    if torch.cuda.is_available():
        print(f"[INFO] CUDA device count : {torch.cuda.device_count()}")
        print(f"[INFO] Current device    : {torch.cuda.current_device()}")
        print(f"[INFO] Device name       : {torch.cuda.get_device_name()}")
    else:
        print("[WARNING] CUDA is not available. Training will run on CPU.")

    # run_meta.json 저장
    run_meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(cfg.data_root),
        "train_ann": str(train_ann),
        "val_ann": str(val_ann),
        "output_dir": str(output_dir),
        "dataset_tag": dataset_tag,
        "model_size": cfg.size,
        "resolution": cfg.resolution,
        "num_queries_requested": cfg.num_queries,
        "num_queries_effective": effective_num_queries,
        "num_select_requested": cfg.num_select,
        "num_select_effective": effective_num_select,
        "group_detr": cfg.group_detr,
        "tune_mode": cfg.tune_mode,
        "freeze_backbone": cfg.freeze_backbone,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "num_workers": cfg.num_workers,
        "effective_batch": eff_batch,
        "lr": cfg.lr,
        "device": cfg.device,
        "amp": cfg.amp,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "early_stopping": cfg.early_stopping,
        "num_classes": cfg.num_classes,
        "class_names": cfg.class_names,
        "seed": cfg.seed,
        "resume": cfg.resume,
        "partial_load": cfg.partial_load,
        "pretrained_ckpt": cfg.pretrained_ckpt,
        "pretrain_weights": cfg.pretrain_weights,
        "wandb_disable": cfg.wandb_disable,
        "wandb_project": cfg.wandb_project,
        "wandb_run": run_name,
        "argv": sys.argv,
        "python": sys.version,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    write_run_meta(output_dir, run_meta)

    # wandb 사용 가능 여부 점검
    use_wandb = not cfg.wandb_disable
    if use_wandb:
        try:
            import wandb as _wandb
            use_wandb = callable(getattr(_wandb, "init", None))
        except Exception:
            use_wandb = False
        if not use_wandb:
            print("[WARN] wandb 사용 불가. wandb를 끕니다. 로그만 로컬에 남습니다.")

    # weights / initialization mode
    if cfg.partial_load and not cfg.pretrained_ckpt:
        raise ValueError("--partial-load를 켰으면 --pretrained-ckpt가 필요합니다.")

    disable_pretrain_weights = False
    pretrain_w: Optional[str] = None
    capacity_ckpt_path: Optional[str] = None

    if cfg.partial_load:
        cfg.pretrained_ckpt = ensure_pretrained_weights(cfg.pretrained_ckpt, redownload=False)
        assert cfg.pretrained_ckpt is not None
        if not os.path.exists(cfg.pretrained_ckpt):
            raise FileNotFoundError(f"pretrained checkpoint not found: {cfg.pretrained_ckpt}")
        disable_pretrain_weights = True
        init_mode = "partial-load"
        capacity_ckpt_path = cfg.pretrained_ckpt
    else:
        explicit_scratch = (
            cfg.pretrain_weights is not None
            and str(cfg.pretrain_weights).strip().lower() in {"", "none", "null"}
        )
        if explicit_scratch:
            disable_pretrain_weights = True
            init_mode = "scratch"
            capacity_ckpt_path = None
        else:
            # None means: use wrapper default pretrained checkpoint for this size.
            pretrain_w = ensure_pretrained_weights(cfg.pretrain_weights, redownload=False)
            init_mode = "pretrained"
            capacity_ckpt_path = pretrain_w if pretrain_w is not None else SIZE_TO_PRETRAIN_WEIGHT_NAME[cfg.size]

    print(f"[INFO] init_mode               : {init_mode}")
    if pretrain_w is not None:
        print(f"[INFO] pretrain_weights(resolved): {pretrain_w}")
    if cfg.partial_load:
        print(f"[INFO] partial_load_ckpt(resolved): {cfg.pretrained_ckpt}")

    # Capacity check for checkpoint-backed initialization
    desired_slots = int(effective_num_queries) * int(cfg.group_detr)
    inferred_capacity = None
    if capacity_ckpt_path:
        cap_info = inspect_query_slots_from_checkpoint(capacity_ckpt_path)
        if cap_info["ok"]:
            inferred_capacity = int(cap_info["total_query_slots"])
            print(
                f"[INFO] checkpoint query slots : {inferred_capacity} "
                f"(desired={desired_slots} from num_queries={effective_num_queries}, group_detr={cfg.group_detr})"
            )
            if (init_mode == "pretrained") and desired_slots > inferred_capacity:
                raise ValueError(
                    "Requested num_queries/group_detr exceeds pretrained checkpoint capacity.\n"
                    f"  checkpoint: {capacity_ckpt_path}\n"
                    f"  available slots: {inferred_capacity}\n"
                    f"  desired slots  : {desired_slots} (num_queries={effective_num_queries}, group_detr={cfg.group_detr})\n"
                    "Use one of:\n"
                    "  1) smaller num_queries\n"
                    "  2) --partial-load --pretrained-ckpt <ckpt>\n"
                    "  3) --pretrain-weights none (scratch)"
                )
        else:
            print(f"[WARN] Could not inspect checkpoint capacity: {capacity_ckpt_path} ({cap_info['error']})")

    run_meta["init_mode"] = init_mode
    run_meta["pretrain_weights_resolved"] = pretrain_w
    run_meta["pretrained_ckpt_resolved"] = cfg.pretrained_ckpt
    run_meta["query_slots_desired"] = desired_slots
    run_meta["query_slots_inferred_capacity"] = inferred_capacity
    write_run_meta(output_dir, run_meta)

    if cfg.dry_run:
        print("[INFO] --dry-run set: exiting before model build/train.")
        return

    # 모델 생성
    model = build_model(
        cfg.size,
        cfg.num_classes,
        cfg.class_names,
        num_queries=effective_num_queries,
        num_select=effective_num_select,
        group_detr=cfg.group_detr,
        pretrain_weights=pretrain_w,
        disable_pretrain_weights=disable_pretrain_weights,
    )

    try:
        # partial load: 호환 텐서만 주입
        torch_module = find_inner_torch_module(model)

        if cfg.partial_load:
            load_compatible_weights_into_torch_module(torch_module, cfg.pretrained_ckpt, verbose=True)

        # v5: finetune policy 적용 (freeze/unfreeze)
        finetune_summary = apply_finetune_policy(
            torch_module,
            tune_mode=cfg.tune_mode,
            freeze_backbone=cfg.freeze_backbone,
        )
        print_finetune_policy_summary(finetune_summary)
        run_meta["finetune_policy"] = finetune_summary
        write_run_meta(output_dir, run_meta)

        # 학습 호출
        call_train(
            model,
            dataset_dir=str(cfg.data_root),
            output_dir=str(output_dir),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            grad_accum_steps=cfg.grad_accum_steps,
            num_workers=cfg.num_workers,
            lr=cfg.lr,
            device=cfg.device,
            resolution=cfg.resolution,
            gradient_checkpointing=cfg.gradient_checkpointing,
            amp=cfg.amp,
            num_select=effective_num_select,
            group_detr=cfg.group_detr,
            early_stopping=cfg.early_stopping,
            resume=cfg.resume,
            wandb=use_wandb,
            project=cfg.wandb_project if use_wandb else None,
            run=run_name if use_wandb else None,
        )

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
