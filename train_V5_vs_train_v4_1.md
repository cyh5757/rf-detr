# train_V5.py vs train_v4_1.py 비교

## 한 줄 요약
- `train_v4_1.py`: 모델 생성 후 **모든 파라미터를 기본 학습 상태**로 두고 `train()` 호출
- `train_V5.py`: 모델 생성 후 **명시적 freeze/unfreeze 정책**을 적용하고, 기본값은 `backbone freeze + query/decoder/head 학습`

## 기존 train_v4_1.py는 어떻게 했나
- 인자에는 `num_queries`, `num_select`, `partial_load`, `pretrain_weights`, `gradient_checkpointing`, `amp` 등이 있음: `train_v4_1.py:536`, `train_v4_1.py:543`
- 하지만 `tune_mode`/`freeze_backbone` 같은 파라미터 선택 학습 인자는 없음: `train_v4_1.py:536`
- 모델 생성 후 partial load가 있으면 로드만 하고, 별도 `requires_grad` 제어 없이 바로 `call_train()`로 진입: `train_v4_1.py:814`, `train_v4_1.py:826`, `train_v4_1.py:833`
- 즉, 실질적으로 optimizer에는 `requires_grad=True`인 기본 파라미터가 모두 들어감

## train_V5.py에서 바뀐 점
- 파인튜닝 모드 상수/키워드 추가: `train_V5.py:29`, `train_V5.py:38`, `train_V5.py:43`
- `apply_finetune_policy()` 추가: 파라미터 이름 기반으로 `requires_grad`를 명시적으로 설정: `train_V5.py:466`
- `print_finetune_policy_summary()` 추가: 학습 전 trainable/frozen 통계를 로그로 출력: `train_V5.py:516`
- 신규 CLI 인자:
  - `--tune-mode {query_only,query_decoder_heads,full}`: `train_V5.py:645`
  - `--freeze-backbone/--no-freeze-backbone` (기본 freeze): `train_V5.py:656`, `train_V5.py:660`
- `run_meta.json`에 정책 정보 저장 (`tune_mode`, `freeze_backbone`, `finetune_policy`): `train_V5.py:837`, `train_V5.py:976`

## V5 기본 정책(요청하신 VRAM 절감 관점)
- 기본 모드: `query_decoder_heads`
- 기본 backbone 처리: `freeze_backbone=True`
- 따라서 학습 대상은 주로 아래 이름 패턴
  - `query_feat`, `refpoint_embed`
  - `transformer.decoder`
  - `class_embed`, `bbox_embed`
  - `segmentation_head`
  - `transformer.enc_out_class_embed`, `transformer.enc_out_bbox_embed`
- 반대로 이름에 `backbone`이 포함된 파라미터는 freeze

## 모드별 동작
- `query_only`
  - query 관련 임베딩(`query_feat`, `refpoint_embed`) 위주 최소 학습
  - VRAM 절감은 가장 유리하지만 성능 리스크는 큼
- `query_decoder_heads` (기본)
  - query + decoder + detection/segmentation head 학습
  - 성능/VRAM 균형점으로 설계
- `full`
  - 전체 학습(단, `--freeze-backbone`를 유지하면 backbone만 제외 가능)

## 실행 예시
```bash
# 권장 시작점: backbone freeze + query/decoder/head
python train_V5.py \
  --size M \
  --num-queries 300 \
  --num-select 300 \
  --tune-mode query_decoder_heads \
  --freeze-backbone \
  --amp \
  --gradient_checkpointing
```

```bash
# 가장 메모리 절약: query only
python train_V5.py \
  --tune-mode query_only \
  --freeze-backbone
```

```bash
# 성능 우선: 거의 전체 학습
python train_V5.py \
  --tune-mode full \
  --no-freeze-backbone
```


python train_v6.py \
  --size 2XL \
  --num-queries 450 --num-select 450 --group-detr 13 \
  --partial-load \
  --pretrained-ckpt /home/mbd1234/rf-detr/rf-detr-seg-xxlarge.pt \
  --batch-size 1 --grad-accum-steps 2 \
  --amp --gradient_checkpointing

python train_v6.py \
  --size 2XL \
  --num-queries 600 --num-select 600 --group-detr 13 \
  --freeze-all-but-queries --unfreeze-head \
  --batch-size 1 --grad-accum-steps 2 \
  --amp --gradient_checkpointing

python train_v6.py \
  --size 2XL \
  --num-queries 600 --num-select 600 --group-detr 13 \
  --freeze-all-but-queries --unfreeze-head \
  --unfreeze-regex "decoder" "transformer" \
  --batch-size 1 --grad-accum-steps 2 \
  --amp --gradient_checkpointing

python train_v6.py \
  --size 2XL \
  --num-queries 600 --num-select 600 --group-detr 13 \
  --freeze-all-but-queries --unfreeze-head \
  --lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 \
  --lora-target-regex "(decoder|transformer|attn|attention)" \
  --batch-size 1 --grad-accum-steps 2 \
  --amp --gradient_checkpointing


# 쿼리만 학습습
python train_v6.py \
  --size 2XL \
  --num-queries 450 --num-select 450 --group-detr 13 \
  --partial-load \
  --pretrained-ckpt /home/mbd1234/rf-detr/outputs/cell_opti_10/2026_02_12_1135__cell_opti_10__Seg2XL_768/checkpoint_best_ema.pth \
  --freeze-all-but-queries \
  --epochs 150 \
  --batch-size 1 --grad-accum-steps 2 \
  --amp --gradient_checkpointing

# 쿼리 + head 학습
python train_v6.py \
  --size 2XL \
  --num-queries 600 --num-select 600 --group-detr 13 \
  --partial-load \
  --pretrained-ckpt /home/mbd1234/rf-detr/outputs/cell_opti_10/2026_02_12_1135__cell_opti_10__Seg2XL_768/checkpoint_best_ema.pth \
  --freeze-all-but-queries --unfreeze-head \
  --epochs 150 \
  --batch-size 1 --grad-accum-steps 2 \
  --amp --gradient_checkpointing


# patial + LoRA + 쿼리 헤드 freeze
python train_v6.py \
  --size 2XL \
  --num-queries 450 --num-select 450 --group-detr 13 \
  --partial-load \
  --pretrained-ckpt /home/mbd1234/rf-detr/outputs/cell_opti_10/2026_02_12_1135__cell_opti_10__Seg2XL_768/checkpoint_best_ema.pth \
  --freeze-all-but-queries --unfreeze-head \
  --lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 \
  --lora-target-regex "(decoder|transformer|attn|attention|self_attn|cross_attn)" \
  --epochs 150 \
  --batch-size 1 --grad-accum-steps 2 \
  --amp --gradient_checkpointing


python train_v6.py \
  --size 2XL \
  --num-queries 350 --num-select 350 --group-detr 13 \
  --partial-load \
  --pretrained-ckpt /home/mbd1234/rf-detr/outputs/cell_opti_10/2026_02_12_1135__cell_opti_10__Seg2XL_768/checkpoint_best_ema.pth \
  --freeze-all-but-queries --unfreeze-head \
  --epochs 150 \
  --batch-size 1 --grad-accum-steps 2 \
  --amp --gradient_checkpointing \
  --alloc-expandable-segments \
  --resolution 640