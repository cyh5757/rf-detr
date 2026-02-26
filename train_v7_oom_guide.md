# train_v7 OOM 대응 가이드 (num_queries * group_detr 고정)

## 목표
- `num_queries * group_detr`는 줄이지 않음
- 학습 OOM을 줄이는 방법을 `train_v7` 코드에 반영

## 결론 요약
- 양자화/가지치기만으로는 **학습 OOM 해결 가능성이 낮음**
- 이유: 현재 병목은 가중치(weight)보다 **activation / attention / matcher cost matrix**
- 따라서 `train_v7`에서는 **학습 그래프 메모리와 matcher 메모리**를 줄이는 방향이 핵심

## 왜 양자화/가지치기만으로 부족한가
1. 양자화(PTQ/INT8)는 주로 추론 최적화
- 학습 중 activation과 autograd 메모리는 크게 남아있음

2. QAT는 학습 오버헤드가 생길 수 있음
- fake quant 연산 추가로 메모리 이점이 제한적

3. 비구조적 가지치기는 dense 연산 기준 VRAM 절감이 작음
- 0이 늘어도 텐서 shape가 같으면 메모리 거의 동일

## train_v7에서 우선 적용할 방법 (효과 큰 순서)
1. Matcher cost matrix chunking
- Hungarian matching 전에 `num_queries_total`을 chunk로 나눠서 비용 계산
- peak VRAM과 CPU RAM 모두 감소
- 대상 파일: `src/rfdetr/models/matcher.py`

2. Decoder activation checkpointing 강제
- 현재의 global checkpointing 외에 decoder block 단위 checkpoint 적용
- 대상 파일: `src/rfdetr/models/transformer.py`

3. Eval 메모리 분리/완화
- eval 주기 늘리기(예: 매 epoch -> 2~5 epoch)
- EMA eval 필요 없으면 비활성
- 이미 추가한 `--profile-vram`으로 eval peak 확인

4. 선택적 precision 최적화
- `--amp` 유지
- fp16/bf16 정책 고정, 불필요한 float32 승격 최소화

5. 옵티마이저/상태 메모리 최소화
- query-only 학습 유지
- weight decay 대상 최소화(필요 시)

## train_v7 권장 CLI 프로필 (예시)
> 아래는 `num_queries*group_detr` 유지 전제에서, 코드 최적화가 들어간 `train_v7.py`를 가정한 예시

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
  --amp \
  --gradient_checkpointing \
  --alloc-expandable-segments \
  --profile-vram
```

## train_v7 구현 상태
- [x] `train_v6.py`를 `train_v7.py`로 분기
- [x] strict query-only 정책 유지 (`refpoint_embed.weight`, `query_feat.weight`만 trainable)
- [x] matcher chunk 옵션 추가: `--matcher-chunk-size`
- [x] eval 주기 옵션 추가: `--eval-interval`
- [x] `run_meta.json`에 메모리 최적화 설정 기록

## 디버깅 루틴
1. `--profile-vram`으로 학습/평가 시점 peak 확인
2. OOM이 train step에서 나면
- matcher chunk size를 더 작게
- decoder checkpointing 적용 여부 확인
3. OOM이 eval에서 나면
- eval interval 증가
- EMA eval 비활성

## 한 줄 가이드
- `num_queries * group_detr`를 고정한다면, `train_v7`의 핵심은 **양자화/가지치기보다 matcher/decoder 메모리 최적화**입니다.
