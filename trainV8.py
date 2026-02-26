#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RF-DETR Segmentation Training Script (cell_opti_10, trainV8)

- 모델 사이즈(N/S/M/L/XL/2XL)에 따른 권장 resolution 자동 적용
- resolution override를 실제 학습 입력 해상도로 적용
- amp / gradient_checkpointing 옵션 추가
- rfdetr의 SegmentationTrainConfig + ModelConfig 기반으로 train() 인자 필터링
- COCO 레이아웃 + segmentation sanity check 옵션
- output_dir: timestamp + dataset_tag + model_tag 포함 (요청사항)
- wandb: 스크립트에서 init하지 않음. model.train(wandb=..., project=..., run=...) 로 전달하면
        rfdetr 내부 MetricsWandBSink에서 한 번만 init (환경/버전에 따라)
- 학습 종료 후 GPU 메모리 정리
- run_meta.json 저장 (재현성/관리용)

실행 예:
  cd /home/mbd1234/rf-detr
  python trainV8.py --size M --resolution 480 --epochs 200 --batch-size 2 --grad-accum-steps 2 --sanity-check

메모리 부족 시:
  python trainV8.py --size S --batch-size 1 --grad-accum-steps 4 --gradient-checkpointing --amp
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import weakref
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    """선택한 모델 크기에 대해 해상도 배수 조건을 확인하고 pos-encoding size를 반환."""
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
# 4) 데이터 검증
# -----------------------------
def assert_coco_layout(data_root: Path) -> tuple[Path, Path]:
    train_ann = data_root / "train" / "_annotations.coco.json"
    valid_ann = data_root / "valid" / "_annotations.coco.json"
    if not train_ann.exists():
        raise FileNotFoundError(f"없음: {train_ann}")
    if not valid_ann.exists():
        # val 폴더 fallback
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
            # polygon: [x1,y1,...] 또는 [[x1,y1,...], ...]
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
# 5) 모델 생성 (rfdetr.detr / rfdetr.config 정의된 클래스명만 사용)
# -----------------------------
def build_model(
    size: str,
    num_classes: int,
    class_names: list[str],
    resolution: int,
    amp: bool,
    gradient_checkpointing: bool,
):
    import rfdetr  # 로컬 import

    size = size.upper()
    patch_size, _, positional_encoding_size = validate_resolution(size, resolution)

    # (detr.py 기준) 세그멘테이션 클래스명
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


# -----------------------------
# 6) train() 호출: SegmentationTrainConfig + ModelConfig 키 전달
# -----------------------------
def call_train(model, **kwargs):
    """
    rfdetr: model.train(**kwargs) -> get_train_config(**kwargs) -> SegmentationTrainConfig.

    - SegmentationTrainConfig에 있는 학습 키 + ModelConfig에 있는 모델 키만 전달.
    - 이 방식으로 resolution/amp/gradient_checkpointing도 실제 전달 가능.
    """
    from rfdetr.config import ModelConfig, SegmentationTrainConfig

    allowed = set(SegmentationTrainConfig.model_fields.keys()) | set(ModelConfig.model_fields.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    dropped = [k for k, v in kwargs.items() if k not in allowed and v is not None]
    if dropped:
        print(f"[WARN] Dropped unsupported train kwargs: {sorted(dropped)}")
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
# 8) run meta 저장 (재현성/관리)
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


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="RF-DETR Segmentation training (cell_opti_10)")

    p.add_argument(
        "--data-root",
        type=str,
        default="/home/mbd1234/data/Optiresolve_result_total_20260223/total_data_refine_1024",
    )
    # outputs_root는 상위 폴더만 지정하고, 내부에서 dataset_tag 하위로 정리
    p.add_argument(
        "--outputs-root",
        type=str,
        default="/home/mbd1234/rf-detr/outputs",
        help="outputs 상위 루트. 실제 저장은 <outputs-root>/<dataset_tag>/... 로 생성",
    )

    p.add_argument("--size", type=str, default="2XL", choices=list(SIZE_TO_RESOLUTION.keys()))
    p.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="미지정 시 size 권장값 자동 적용. 지정 시 실제 학습 해상도로 적용",
    )

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. 워커 크래시 시 0/1 권장")
    p.add_argument("--pad-to-square", dest="pad_to_square", action="store_true", default=True,
                   help="원본 비율 유지 후 우하단 패딩으로 정사각형 맞춤")
    p.add_argument("--no-pad-to-square", dest="pad_to_square", action="store_false",
                   help="기존 동작(정사각 강제 리사이즈, 왜곡 가능)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--amp", dest="amp", action="store_true", default=True, help="AMP mixed precision 사용")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="AMP 비활성화")
    p.add_argument(
        "--gradient-checkpointing",
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Gradient checkpointing 활성화 (메모리 절감, 속도 저하 가능)",
    )

    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--class-names", type=str, nargs="+", default=["background", "cell"])

    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sanity-check", action="store_true")

    # wandb는 script에서 init하지 않지만, rfdetr 내부에서 받을 수 있으니 전달값만 받는다
    p.add_argument("--wandb-project", type=str, default="cell_rfdetr")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-disable", action="store_true")

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
    )


# -----------------------------
# 10) Main
# -----------------------------
def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    patch_size, num_windows, positional_encoding_size = validate_resolution(cfg.size, cfg.resolution)

    train_ann, val_ann = assert_coco_layout(cfg.data_root)
    if cfg.sanity_check:
        quick_segmentation_sanity_check(train_ann)

    dataset_tag = cfg.data_root.name  # e.g. cell_opti_10
    model_tag = f"Seg{cfg.size}_{cfg.resolution}"
    if cfg.gradient_checkpointing:
        model_tag += "__GC"
    if cfg.amp:
        model_tag += "__AMP"
    prefix = f"{dataset_tag}__{model_tag}"

    # outputs: <outputs_root>/<dataset_tag>/<timestamp__prefix>
    outputs_base = cfg.outputs_root / dataset_tag
    output_dir = create_output_dir(outputs_base, prefix=prefix)

    run_name = cfg.wandb_run_name or f"{model_tag}__{output_dir.name}"
    eff_batch = cfg.batch_size * cfg.grad_accum_steps

    print("====================================================")
    print("[INFO] RF-DETR Seg Training Start")
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
    print(f"[INFO] epochs            : {cfg.epochs}")
    print(f"[INFO] batch_size        : {cfg.batch_size}")
    print(f"[INFO] grad_accum_steps  : {cfg.grad_accum_steps} (effective={eff_batch})")
    print(f"[INFO] num_workers       : {cfg.num_workers}")
    print(f"[INFO] pad_to_square     : {cfg.pad_to_square}")
    print(f"[INFO] lr                : {cfg.lr}")
    print(f"[INFO] amp               : {cfg.amp}")
    print(f"[INFO] gradient_ckpt     : {cfg.gradient_checkpointing}")
    print(f"[INFO] patch_size        : {patch_size}")
    print(f"[INFO] num_windows       : {num_windows}")
    print(f"[INFO] pos_enc_size      : {positional_encoding_size}")
    print(f"[INFO] early_stopping    : {cfg.early_stopping}")
    print(f"[INFO] num_classes       : {cfg.num_classes}")
    print(f"[INFO] class_names       : {cfg.class_names}")
    print(f"[INFO] seed              : {cfg.seed}")
    if cfg.resume:
        print(f"[INFO] resume            : {cfg.resume}")
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

    # run_meta.json 저장 (학습 전에 먼저 남겨두면 실패 시에도 흔적이 남음)
    run_meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(cfg.data_root),
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
        "wandb_disable": cfg.wandb_disable,
        "wandb_project": cfg.wandb_project,
        "wandb_run": run_name,
        "argv": sys.argv,
        "python": sys.version,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    write_run_meta(output_dir, run_meta)

    # wandb 사용 가능 여부: 로컬 wandb.py 섀도잉 또는 깨진 설치 시 AttributeError 방지
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

    # 모델 생성
    model = build_model(
        cfg.size,
        cfg.num_classes,
        cfg.class_names,
        resolution=cfg.resolution,
        amp=cfg.amp,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    train_kwargs = dict(
        dataset_dir=str(cfg.data_root),
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

    try:
        # 학습: SegmentationTrainConfig + ModelConfig 키만 전달
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
