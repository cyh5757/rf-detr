# train_v6.py CLI 사용법 (Strict Query-Only)

## 목적
`train_v6.py`는 **`num_queries`와 직접 연결된 query slot weight만 학습**하도록 고정되어 있습니다.

학습되는 파라미터:
- `refpoint_embed.weight`
- `query_feat.weight`

그 외 파라미터는 모두 `freeze` 됩니다.

## 기본 실행
```bash
python /home/mbd1234/rf-detr/train_v6.py \
  --size 2XL \
  --num-queries 450 \
  --num-select 450 \
  --group-detr 13 \
  --batch-size 1 \
  --grad-accum-steps 2 \
  --epochs 150 \
  --amp \
  --gradient_checkpointing
```

## 부분 로딩(권장)
기존 체크포인트에서 shape가 맞는 텐서만 부분 로딩합니다.

```bash
python /home/mbd1234/rf-detr/train_v6.py \
  --size 2XL \
  --num-queries 450 \
  --num-select 450 \
  --group-detr 13 \
  --partial-load \
  --pretrained-ckpt /home/mbd1234/rf-detr/outputs/cell_opti_10/2026_02_12_1135__cell_opti_10__Seg2XL_768/checkpoint_best_ema.pth \
  --batch-size 1 \
  --grad-accum-steps 2 \
  --epochs 150 \
  --amp \
  --gradient_checkpointing
```

## 사전 점검용 dry-run
실제 학습 없이 설정/용량 체크만 수행합니다.

```bash
python /home/mbd1234/rf-detr/train_v6.py \
  --size 2XL \
  --num-queries 450 \
  --num-select 450 \
  --group-detr 13 \
  --dry-run
```

## query 슬롯 용량 확인
체크포인트가 지원 가능한 최대 `num_queries`를 확인할 때 사용합니다.

```bash
python /home/mbd1234/rf-detr/train_v6.py \
  --group-detr 13 \
  --inspect-ckpts /home/mbd1234/rf-detr/rf-detr-seg-xxlarge.pt
```

## 자주 쓰는 인자
- `--size {N,S,M,L,XL,2XL}`: 모델 크기
- `--num-queries`: query 개수
- `--num-select`: 후처리 top-k (반드시 `num_select <= num_queries`)
- `--group-detr`: grouped query 배수 계산에 사용
- `--partial-load --pretrained-ckpt <path>`: shape 호환 텐서만 로딩
- `--amp`: mixed precision
- `--gradient_checkpointing`: 메모리 절감
- `--alloc-expandable-segments`: CUDA 메모리 파편화 완화
- `--profile-vram`: eval 단계의 peak/current VRAM 로그 출력

## 출력
결과는 `--outputs-root` 하위에 저장됩니다.
- `checkpoint_best_total.pth`
- `checkpoint_best_ema.pth`
- `run_meta.json`

## OOM-safe 프로필 (num_queries=450 + partial-load)
아래는 VRAM 터짐을 줄이기 위한 시작점입니다.

```bash
python /home/mbd1234/rf-detr/train_v6.py \
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
  --alloc-expandable-segments
```

### OOM이 나면 줄이는 순서
1. `--group-detr 8 -> 6`
2. `--resolution 640 -> 576`
3. 그래도 OOM이면 `--size XL`

### 참고
- `partial-load`는 주로 초기화 방식이며, VRAM 사용량은 주로 `size`, `resolution`, `group-detr`, `batch-size`에 의해 결정됩니다.
- 현재 `train_v6.py`는 strict query-only 정책이므로 query weight(`refpoint_embed.weight`, `query_feat.weight`)만 학습됩니다.

## Eval VRAM 확인 방법
학습 중 각 eval 시점의 VRAM을 보고 싶으면 `--profile-vram`을 추가합니다.

```bash
python train_v6.py \
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
  --profile-vram
```

로그 예시:
- `[VRAM][EVAL] current_alloc=...MB current_reserved=...MB peak_alloc=...MB peak_reserved=...MB`
