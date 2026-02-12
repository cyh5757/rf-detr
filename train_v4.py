#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RF-DETR Segmentation Training Script (cell_opti_10) - v4

핵심 기능 (v4 추가):
- num_queries 변경 지원 (--num-queries)
- num_queries 변경 시 pretrained weight 로딩 mismatch를 피하기 위한 partial weight loading (--partial-load, --pretrained-ckpt)
- 모델 클래스가 pretrain_weights 인자를 지원하면 자동 로딩을 강제로 끔(pretrain_weights=None) 후,
  원하는 체크포인트를 "호환 텐서만" 주입하는 방식으로 pretrained 활용

관련 이슈:
- num_queries 변경 시 pretrained mismatch 예시: https://github.com/roboflow/rf-detr/issues/419
- partial load 아이디어/코드: https://github.com/roboflow/rf-detr/issues/293
"""

from __future__ import annotations

import argparse
import gc
import inspect
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


# -----------------------------
# 1) Size -> Resolution mapping (사용자가 제공한 표 그대로)
# -----------------------------
SIZE_TO_RESOLUTION = {
    "N": 312,
    "S": 384,
    "M": 432,
    "L": 504,
    "XL": 624,
    "2XL": 768,
}


# -----------------------------
# 2) 유틸: output_dir 생성 (timestamp + tags)
# -----------------------------
def create_output_dir(base_dir: str | Path, prefix: str) -> Path:
    """
    base_dir 아래에 현재 날짜/시간 + prefix 기반 폴더 생성.
    예) base_dir='.../outputs/seg_cell_opti_10'
        -> '.../outputs/seg_cell_opti_10/2026_02_12_1040__cell_opti_10__SegM_432'
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    out_dir = base_dir / f"{timestamp}__{prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# -----------------------------
# 3) 유틸: GPU 메모리 정리
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
    """
    체크포인트에서 module.state_dict()와 key/shape가 완전히 일치하는 텐서만 로드.
    num_queries 변경으로 인해 query/refpoint 관련 텐서 mismatch가 나도 안전하게 우회 가능.

    참고 아이디어: https://github.com/roboflow/rf-detr/issues/293
    mismatch 상황 예: https://github.com/roboflow/rf-detr/issues/419
    """
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

        # 너무 길어지지 않게 샘플 출력
        skipped_keys = [k for k in model_sd.keys() if k not in compatible]
        if skipped_keys:
            print(f"[PARTIAL LOAD] Example skipped keys (up to 12): {skipped_keys[:12]}")

    return loaded, total


def find_inner_torch_module(model) -> torch.nn.Module:
    """
    rfdetr wrapper 내부 torch.nn.Module 위치가 버전마다 달라질 수 있으므로
    흔한 경로들을 순서대로 탐색한다.
    """
    candidates = [
        "model.model.model",  # 이슈 #293 예시에서 사용된 경로
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
        "내부 torch.nn.Module 경로를 찾지 못했습니다. "
        "model 구조를 확인해야 합니다(버전 차이 가능)."
    )


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
    """segmentation 폴리곤이 실제로 들어있는지(대충이라도) 확인."""
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


# -----------------------------
# 5) 모델 생성
# -----------------------------
def build_model(size: str, num_classes: int, class_names: list[str], num_queries: Optional[int] = None):
    import rfdetr  # 로컬 import

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
            sig = inspect.signature(cls)
            kwargs = {}

            if "num_classes" in sig.parameters:
                kwargs["num_classes"] = num_classes
            if "class_names" in sig.parameters:
                kwargs["class_names"] = class_names

            # num_queries 지원 시 주입
            if num_queries is not None and "num_queries" in sig.parameters:
                kwargs["num_queries"] = num_queries

            # 자동 pretrained 로딩 방지(지원할 경우)
            # num_queries 바꾸면 pretrained 로딩에서 shape mismatch 터질 수 있음 (#419)
            if "pretrain_weights" in sig.parameters:
                kwargs["pretrain_weights"] = None

            return cls(**kwargs)
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"모델 클래스를 찾았지만 생성에 실패. candidates={candidates}, last_err={last_err}"
    )


# -----------------------------
# 6) train() 호출: SegmentationTrainConfig(model_fields) 키만 전달
# -----------------------------
def call_train(model, **kwargs):
    """
    rfdetr: model.train(**kwargs) -> get_train_config(**kwargs) -> SegmentationTrainConfig.

    - SegmentationTrainConfig에 없는 키는 전달하지 않음.
    """
    from rfdetr.config import SegmentationTrainConfig

    allowed = set(SegmentationTrainConfig.model_fields.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}

    # 사용자가 resolution을 태깅만 하는 게 아니라 실제 반영하고 싶을 수도 있음.
    # 다만 여기서는 allowed에 있으면 자동으로 전달되도록만 둔다.
    return model.train(**filtered)


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
    lr: float
    num_classes: int
    class_names: list[str]
    early_stopping: bool
    resume: Optional[str]
    seed: int
    sanity_check: bool
    wandb_project: Optional[str]
    wandb_run_name: Optional[str]
    wandb_disable: bool

    # v4 추가
    num_queries: Optional[int]
    partial_load: bool
    pretrained_ckpt: Optional[str]


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="RF-DETR Segmentation training (cell_opti_10) - v4")

    p.add_argument(
        "--data-root",
        type=str,
        default="/home/mbd1234/data/Optiresolve_result_20260212/cell_opti_10",
    )
    p.add_argument(
        "--outputs-root",
        type=str,
        default="/home/mbd1234/rf-detr/outputs",
        help="outputs 상위 루트. 실제 저장은 <outputs-root>/<dataset_tag>/... 로 생성",
    )

    p.add_argument("--size", type=str, default="M", choices=list(SIZE_TO_RESOLUTION.keys()))
    p.add_argument("--resolution", type=int, default=None, help="미지정 시 size에 따른 권장값 자동 적용(태깅/옵션용)")

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)

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

    # v4: num_queries / partial load
    p.add_argument("--num-queries", type=int, default=None, help="DETR queries 수(상한 증가 목적)")
    p.add_argument("--partial-load", action="store_true", help="shape 동일한 pretrained weight만 부분 로딩")
    p.add_argument("--pretrained-ckpt", type=str, default=None, help="부분 로딩할 pretrained checkpoint(.pth) 경로")

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
        lr=a.lr,
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
        partial_load=bool(a.partial_load),
        pretrained_ckpt=a.pretrained_ckpt,
    )


# -----------------------------
# 10) Main
# -----------------------------
def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    train_ann, val_ann = assert_coco_layout(cfg.data_root)
    if cfg.sanity_check:
        quick_segmentation_sanity_check(train_ann)

    dataset_tag = cfg.data_root.name  # e.g. cell_opti_10
    model_tag = f"Seg{cfg.size}_{cfg.resolution}"
    if cfg.num_queries is not None:
        model_tag = f"{model_tag}__Q{cfg.num_queries}"
    prefix = f"{dataset_tag}__{model_tag}"

    outputs_base = cfg.outputs_root / dataset_tag
    output_dir = create_output_dir(outputs_base, prefix=prefix)

    run_name = cfg.wandb_run_name or f"{model_tag}__{output_dir.name}"
    eff_batch = cfg.batch_size * cfg.grad_accum_steps

    print("====================================================")
    print("[INFO] RF-DETR Seg Training Start (v4)")
    print(f"[INFO] data_root         : {cfg.data_root}")
    print(f"[INFO] train_ann         : {train_ann}")
    print(f"[INFO] val_ann           : {val_ann}")
    print(f"[INFO] output_dir        : {output_dir}")
    print(f"[INFO] dataset_tag       : {dataset_tag}")
    print(f"[INFO] model_tag         : {model_tag}")
    print(f"[INFO] size              : {cfg.size}")
    print(
        f"[INFO] resolution        : {cfg.resolution} (size mapping default)"
        if cfg.resolution == SIZE_TO_RESOLUTION[cfg.size]
        else f"[INFO] resolution        : {cfg.resolution} (override)"
    )
    print(f"[INFO] num_queries       : {cfg.num_queries}")
    print(f"[INFO] epochs            : {cfg.epochs}")
    print(f"[INFO] batch_size        : {cfg.batch_size}")
    print(f"[INFO] grad_accum_steps  : {cfg.grad_accum_steps} (effective={eff_batch})")
    print(f"[INFO] lr                : {cfg.lr}")
    print(f"[INFO] early_stopping    : {cfg.early_stopping}")
    print(f"[INFO] num_classes       : {cfg.num_classes}")
    print(f"[INFO] class_names       : {cfg.class_names}")
    print(f"[INFO] seed              : {cfg.seed}")
    if cfg.resume:
        print(f"[INFO] resume            : {cfg.resume}")

    print(f"[INFO] partial_load      : {cfg.partial_load}")
    if cfg.partial_load:
        print(f"[INFO] pretrained_ckpt   : {cfg.pretrained_ckpt}")

    print(f"[INFO] wandb             : {'OFF' if cfg.wandb_disable else 'ON (internal)'}")
    if not cfg.wandb_disable:
        print(f"[INFO] wandb_project     : {cfg.wandb_project}")
        print(f"[INFO] wandb_run         : {run_name}")
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
        "resolution_tag": cfg.resolution,
        "num_queries": cfg.num_queries,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "effective_batch": eff_batch,
        "lr": cfg.lr,
        "early_stopping": cfg.early_stopping,
        "num_classes": cfg.num_classes,
        "class_names": cfg.class_names,
        "seed": cfg.seed,
        "resume": cfg.resume,
        "partial_load": cfg.partial_load,
        "pretrained_ckpt": cfg.pretrained_ckpt,
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
    use_wandb = True
    if not cfg.wandb_disable:
        try:
            import wandb as _wandb
            use_wandb = callable(getattr(_wandb, "init", None))
        except Exception:
            pass
        if not use_wandb:
            print("[WARN] wandb 사용 불가(미설치 또는 init 없음). wandb를 끕니다. 로그만 로컬에 남습니다.")
            print("       해결: pip install wandb 또는 프로젝트 내 wandb.py 파일 제거 후 재실행.")

    # partial load 인자 검증
    if cfg.partial_load and not cfg.pretrained_ckpt:
        raise ValueError("--partial-load를 켰으면 --pretrained-ckpt 경로가 필요합니다.")
    if cfg.pretrained_ckpt and not Path(cfg.pretrained_ckpt).exists():
        raise FileNotFoundError(f"pretrained checkpoint not found: {cfg.pretrained_ckpt}")

    # 모델 생성 (num_queries 주입 + 자동 pretrained 로딩 방지(pretrain_weights=None))
    model = build_model(cfg.size, cfg.num_classes, cfg.class_names, num_queries=cfg.num_queries)

    try:
        # partial load: 호환 텐서만 주입
        if cfg.partial_load:
            torch_module = find_inner_torch_module(model)
            load_compatible_weights_into_torch_module(torch_module, cfg.pretrained_ckpt, verbose=True)

        # 학습
        call_train(
            model,
            dataset_dir=str(cfg.data_root),
            output_dir=str(output_dir),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            grad_accum_steps=cfg.grad_accum_steps,
            lr=cfg.lr,
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
