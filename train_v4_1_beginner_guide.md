# train_v4_1.py 초보자 가이드

## 이 문서의 목적
`train_v4_1.py`를 처음 보는 사람이 아래를 이해하도록 정리한 문서입니다.

1. `partial-load`와 `scratch`의 차이
2. 실제로 어떤 파라미터가 학습되는지
3. 언제 어떤 옵션을 쓰면 좋은지

## 먼저 큰 그림
`train_v4_1.py`는 크게 2단계로 동작합니다.

1. 모델을 어떤 방식으로 초기화할지 결정
2. 그 모델을 `model.train(...)`으로 학습

초기화 방식은 3가지입니다.

1. `pretrained` (기본)
2. `partial-load`
3. `scratch`

중요한 포인트:
- `train_v4_1.py`에는 **freeze/unfreeze 로직이 없습니다.**
- 즉, 기본적으로 `requires_grad=True`인 파라미터는 모두 학습됩니다.
- 차이는 "무엇을 학습할지"가 아니라, **"어떤 값으로 학습을 시작할지(초기값)"**입니다.

## 용어 정리
- `pretrained`: 미리 학습된 가중치로 시작
- `scratch`: 랜덤 초기값으로 시작
- `partial-load`: 체크포인트에서 **이름/shape가 맞는 텐서만** 골라서 로드하고 시작

## v4.1에서 실제 분기 로직
`train_v4_1.py` 기준으로 초기화 모드는 다음처럼 정해집니다.

- `--partial-load`를 켜면: `init_mode = "partial-load"`
- 그렇지 않고 `--pretrain-weights none`이면: `init_mode = "scratch"`
- 그 외에는: `init_mode = "pretrained"`

관련 위치:
- 인자 정의: `train_v4_1.py:536`, `train_v4_1.py:543`
- 모드 분기: `train_v4_1.py:749`, `train_v4_1.py:758`
- 학습 호출: `train_v4_1.py:833`

## partial-load vs scratch (핵심 비교)

| 항목 | partial-load | scratch |
|---|---|---|
| 시작 가중치 | 체크포인트에서 shape 맞는 것만 로드 | 전부 랜덤 초기화 |
| 목적 | `num_queries` 변경 등으로 일부 shape mismatch가 있어도 pretrained 이점 최대 활용 | 완전 새로 학습 |
| 수렴 속도 | 보통 더 빠름 | 보통 더 느림 |
| 초기 성능 | 보통 더 높음 | 보통 더 낮음 |
| 안정성 | 데이터 적을 때 유리한 경우 많음 | 데이터 충분하면 가능하지만 시간 필요 |

## "뭘 학습하는가"를 정확히 보면

### 1) partial-load일 때
- 로딩 단계에서:
  - backbone/decoder/head/query 중 **shape가 맞는 텐서**는 pretrained 값으로 시작
  - shape가 안 맞는 텐서(예: query 수 변경으로 달라진 텐서)는 랜덤으로 남음
- 학습 단계에서:
  - **모든 trainable 파라미터가 학습**됨

즉, partial-load는 "일부만 학습"이 아니라 "일부만 로드"입니다.

### 2) scratch일 때
- 로딩 단계에서:
  - pretrained를 쓰지 않으므로 사실상 전체 랜덤 시작
- 학습 단계에서:
  - **모든 trainable 파라미터가 학습**됨

즉, scratch도 학습 대상 자체는 partial-load와 같습니다. 다른 점은 시작점뿐입니다.

## 네트워크별로 보면 (v4.1)
`train_v4_1.py` 자체에서 특정 모듈을 freeze하지 않기 때문에, 아래 영역이 기본적으로 다 학습됩니다.

1. Backbone (DINOv2/ViT + projector)
2. DETR transformer (encoder/decoder 관련 파라미터)
3. Detection head (`class_embed`, `bbox_embed` 계열)
4. Segmentation head
5. Query 관련 임베딩 (`query_feat`, `refpoint_embed`)

단, partial-load에서는 위 중 일부가 pretrained로 초기화되고, scratch에서는 전부 랜덤 초기화된다는 차이만 있습니다.

## partial-load가 특히 필요한 상황
다음 같은 경우 partial-load가 실무에서 자주 쓰입니다.

1. `num_queries`를 바꿨다
2. `group_detr` 설정 변경으로 query 관련 shape가 달라졌다
3. pretrained를 최대한 쓰고 싶은데 shape mismatch 에러를 피하고 싶다

v4.1의 `load_compatible_weights_into_torch_module(...)`가 바로 이 문제를 처리합니다.
- key/shape가 같은 텐서만 골라 로드
- 안 맞는 텐서는 건너뜀

관련 위치:
- 함수: `train_v4_1.py:149`
- 적용: `train_v4_1.py:828`

## 어떤 옵션을 언제 쓰면 좋은가 (초보자 기준)

1. 처음 시작
- 추천: `pretrained` 또는 `partial-load`
- 이유: 더 빨리 안정적으로 수렴할 가능성이 높음

2. `num_queries`를 바꿔서 mismatch가 걱정될 때
- 추천: `--partial-load --pretrained-ckpt <ckpt>`
- 이유: 로드 가능한 부분은 살리고, 충돌은 회피

3. 정말 완전 새로 학습하고 싶을 때
- 추천: `--pretrain-weights none` (scratch)
- 이유: pretrained bias 없이 완전 재학습

## 자주 하는 오해

1. "partial-load면 일부 모듈만 학습하나요?"
- 아닙니다. 일부만 "로드"하고, 학습은 기본적으로 전체 trainable 파라미터 대상입니다.

2. "scratch가 메모리를 줄여주나요?"
- 보통 아닙니다. VRAM은 주로 배치/해상도/모델크기/활성값 저장량이 좌우합니다.

3. "v4.1에서 backbone을 자동으로 얼려주나요?"
- 아닙니다. v4.1에는 그런 로직이 없습니다.

## 참고 실행 예시

### partial-load 예시
```bash
python train_v4_1.py \
  --size M \
  --num-queries 300 \
  --num-select 300 \
  --partial-load \
  --pretrained-ckpt rf-detr-seg-medium.pt
```

### scratch 예시
```bash
python train_v4_1.py \
  --size M \
  --num-queries 300 \
  --num-select 300 \
  --pretrain-weights none
```

## 마지막 정리
- `partial-load` vs `scratch`의 본질은 **초기값 차이**입니다.
- v4.1에서 학습 대상은 기본적으로 동일합니다(명시적 freeze 없음).
- "학습할 모듈을 줄이는 전략"이 필요하면, v5처럼 freeze 정책이 있는 스크립트를 쓰는 것이 맞습니다.
