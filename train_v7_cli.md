# train_v7.py CLI 사용법

## 핵심
- strict query-only: `refpoint_embed.weight`, `query_feat.weight`만 학습
- OOM 제어 옵션 추가:
  - `--matcher-chunk-size`
  - `--eval-interval`
  - `--profile-vram`

## 기본 실행 (OOM 제어 포함)
```bash
python /home/mbd1234/rf-detr/train_v7.py \
  --size 2XL \
  --resolution 640 \
  --num-queries 450 \
  --num-select 450 \
  --group-detr 8 \
  --partial-load \
  --pretrained-ckpt /home/mbd1234/rf-detr/outputs/cell_opti_10/2026_02_12_1135__cell_opti_10__Seg2XL_768/checkpoint_best_ema.pth \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 0 \
  --epochs 150 \
  --amp \
  --gradient_checkpointing \
  --alloc-expandable-segments \
  --matcher-chunk-size 128 \
  --eval-interval 2 \
  --profile-vram
```

## 옵션 설명
- `--oom-safe`: OOM 완화 기본값 자동 적용(단, `num_queries`는 변경하지 않음)
  - 강제/보정 항목: `--amp`, `--gradient_checkpointing`, `--alloc-expandable-segments`, `--num-workers 0`, `--matcher-chunk-size 128`, `--eval-interval 2`
- `--matcher-chunk-size`: Hungarian matcher cost 계산을 row chunk로 분할
  - `0`: chunk 비활성
  - 예: `64`, `128`, `256`
- `--eval-interval`: validation 주기(에폭 단위)
  - 예: `2`면 2 에폭마다 eval
  - 마지막 에폭에서는 항상 eval 수행
- `--profile-vram`: eval 시 VRAM peak/current 로그 출력

## 튜닝 순서 (OOM 시)
1. `--matcher-chunk-size 128 -> 64`
2. `--eval-interval 1 -> 2 -> 5`
3. 그래도 OOM이면 `--resolution` 또는 `--size` 조정

## 참고
- `num_queries * group_detr`가 동일하면, 양자화/가지치기보다 matcher/activation 메모리 최적화가 학습 OOM 완화에 더 직접적입니다.

## 최소 입력으로 OOM-safe 실행
```bash
python /home/mbd1234/rf-detr/train_v7.py \
  --size 2XL \
  --resolution 640 \
  --num-queries 450 \
  --num-select 450 \
  --group-detr 8 \
  --partial-load \
  --pretrained-ckpt /home/mbd1234/rf-detr/outputs/cell_opti_10/2026_02_12_1135__cell_opti_10__Seg2XL_768/checkpoint_best_ema.pth \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --epochs 150 \
  --oom-safe \
  --profile-vram
```
