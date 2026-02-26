# RF-DETR Inference 사용 가이드

작성일: 2026-02-26  
대상 파일: `rf-detr/inference.py`

## 1) 개요
`inference.py`는 RF-DETR 세그멘테이션 체크포인트와 SAHI 슬라이싱을 이용해 이미지를 추론하고, 아래 결과를 저장합니다.

1. 패딩 이미지: `padding/<run_name>/...`
2. 시각화 이미지: `inferenced/<run_name>/...`
3. COCO 어노테이션: `annote/<run_name>/_annotations.coco.json`
4. 요약 파일: `annote/<run_name>/inference_summary.json`

입력 이미지는 중앙 정렬 black square padding 후 추론됩니다.

## 2) 실행 위치와 기본 실행
프로젝트 루트(`/home/mbd1234/rf-detr`)에서 실행하세요.

```bash
cd /home/mbd1234/rf-detr
/home/mbd1234/.new_rfdetr/bin/python inference.py
```

별도 가상환경을 쓴다면 해당 환경의 `python`으로 실행하면 됩니다.

## 3) 출력 폴더 구조
`--run-name`을 지정하지 않으면 자동으로 `<size>__<checkpoint_parent_or_stem>`이 생성됩니다.

예시:

- `annote/XL__2026_02_25_1042__total_data_refine_1024__SegXL_624/_annotations.coco.json`
- `padding/XL__2026_02_25_1042__total_data_refine_1024__SegXL_624/*.jpg`
- `inferenced/XL__2026_02_25_1042__total_data_refine_1024__SegXL_624/*.jpg`

## 4) 자주 쓰는 실행 예시

### 4-1. 기본값으로 전체 실행
```bash
/home/mbd1234/.new_rfdetr/bin/python inference.py
```

### 4-2. 체크포인트/입출력 경로 지정
```bash
/home/mbd1234/.new_rfdetr/bin/python inference.py \
  --checkpoint /path/to/checkpoint_best_ema.pth \
  --image-dir /path/to/images \
  --annote-dir /path/to/annote \
  --padding-dir /path/to/padding \
  --inferenced-dir /path/to/inferenced
```

### 4-3. run_name 고정
```bash
/home/mbd1234/.new_rfdetr/bin/python inference.py \
  --run-name cell_pseudo_v1
```

### 4-4. 50장만 빠른 테스트
```bash
/home/mbd1234/.new_rfdetr/bin/python inference.py \
  --max-images 50 \
  --slice-size 1024 \
  --overlap 0.15
```

### 4-5. JSON만 생성(이미지 저장 비활성화)
```bash
/home/mbd1234/.new_rfdetr/bin/python inference.py \
  --no-save-padded \
  --no-save-visuals
```

### 4-6. 기존 COCO 무시하고 처음부터 재생성
```bash
/home/mbd1234/.new_rfdetr/bin/python inference.py \
  --no-skip-existing
```

### 4-7. 시각화 스타일 제어
```bash
/home/mbd1234/.new_rfdetr/bin/python inference.py \
  --vis-bbox \
  --vis-id \
  --vis-score \
  --vis-contour \
  --vis-fill-mask \
  --vis-alpha 0.30 \
  --vis-bbox-thickness 2 \
  --vis-contour-thickness 2
```

### 4-8. SAHI 후처리 변경
```bash
/home/mbd1234/.new_rfdetr/bin/python inference.py \
  --postprocess-type NMS \
  --postprocess-match-metric IOU \
  --postprocess-match-threshold 0.55
```

## 5) 주요 옵션 정리

### 5-1. 입력/출력
- `--checkpoint`: RF-DETR 체크포인트(.pth/.pt)
- `--image-dir`: 추론할 원본 이미지 폴더
- `--annote-dir`: COCO JSON 저장 베이스 폴더
- `--padding-dir`: 패딩 이미지 저장 베이스 폴더
- `--inferenced-dir`: 시각화 이미지 저장 베이스 폴더
- `--run-name`: 결과 하위 폴더명(미지정 시 자동 생성)

### 5-2. 모델/임계값
- `--size`: `N|S|M|L|XL|2XL`
- `--device`: `cuda`, `cuda:0`, `cpu`, `mps`
- `--confidence-threshold`: 객체 confidence 기준
- `--mask-threshold`: mask 이진화 기준
- `--min-area`: 최소 mask 면적(px)
- `--optimize-inference` / `--no-optimize-inference`: `optimize_for_inference()` 사용 여부
- `--optimize-compile` / `--no-optimize-compile`: TorchScript trace 컴파일 사용 여부
- `--optimize-dtype`: `float16|bfloat16|float32` (속도는 보통 `float16`, VRAM 많이 쓰려면 `float32`)
- `--optimize-batch-size`: compile 시 기준 배치 크기(현재 SAHI 기본 경로는 보통 `1` 권장)

### 5-3. SAHI
- `--slice-size`: 슬라이스 크기
- `--overlap`: 슬라이스 overlap 비율
- `--postprocess-type`: `NMM|GREEDYNMM|NMS|LSNMS`
- `--postprocess-match-metric`: `IOU|IOS`
- `--postprocess-match-threshold`: 병합 기준
- `--postprocess-class-agnostic`: 클래스 무시 병합
- `--perform-standard-pred`: 전체 이미지 추론 추가 실행

### 5-4. 실행 제어
- `--max-images`: 앞에서 N장만 실행
- `--save-padded` / `--no-save-padded`: 패딩 저장 on/off
- `--save-visuals` / `--no-save-visuals`: 시각화 저장 on/off
- `--skip-existing` / `--no-skip-existing`: 기존 COCO 파일 기준 skip on/off
- `--autosave-interval`: N장마다 중간 JSON 저장(0이면 비활성)
- `--verbose-sahi`: `0|1|2`

### 5-5. 시각화 오버레이
- `--vis-bbox` / `--no-vis-bbox`
- `--vis-id` / `--no-vis-id`
- `--vis-score` / `--no-vis-score`
- `--vis-contour` / `--no-vis-contour`
- `--vis-fill-mask` / `--no-vis-fill-mask`
- `--vis-alpha`: mask 투명도(0~1)
- `--vis-contour-thickness`
- `--vis-bbox-thickness`

## 6) 운영 팁

1. 먼저 `--max-images 20~50`으로 품질 확인 후 전체 실행하세요.
2. 속도가 중요하면 `--no-save-padded --no-save-visuals`로 JSON만 먼저 생성하세요.
3. 이어서 실행할 때는 기본값(`--skip-existing`)을 유지하면 이미 처리된 파일은 건너뜁니다.
4. 완전 재생성하려면 `--no-skip-existing`를 사용하세요.

## 7) 에러 체크 포인트

1. `Checkpoint not found`: `--checkpoint` 경로 확인
2. `Image directory not found`: `--image-dir` 경로 확인
3. `No images found`: 이미지 확장자(`.jpg/.jpeg/.png/.bmp/.tif/.tiff`) 확인
4. CUDA 관련 오류: `--device cpu`로 먼저 동작 검증 후 CUDA 환경 점검
