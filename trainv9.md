# trainV9 계획 (Tile-Based Instance Segmentation)

작성일: 2026-02-26  
기준: `trainV8.py`

## 1) 목적

- 인스턴스 세그멘테이션에서 이미지당 객체 수가 매우 많은 경우(예: 1500+)에 발생하는 학습/추론 병목(질의 슬롯/후처리 cap)을 줄인다.
- 원본 이미지 1장 단위 대신, 모델 입력 해상도 기준 타일 단위로 이미지+annotation을 잘라 학습한다.
- 결과적으로 한 샘플(타일) 내 객체 밀도를 낮춰 더 안정적으로 학습시키는 `trainV9` 파이프라인을 만든다.

## 2) 현재 상태 요약 (데이터 밀도)

- `total_data_refine_1024/train` 기준:
  - 이미지 800장, annotation 392,377개
  - 이미지당 annotation: 평균 490, p95 1289, 최대 2017
- `total_data_refine_1024/valid` 기준:
  - 이미지 100장, annotation 44,339개
  - 이미지당 annotation: 평균 443, p95 989, 최대 1613
- 결론: 원본 이미지 단위 학습 시 객체 과밀도가 높아 cap/slot 제한의 영향을 크게 받을 가능성이 높다.

## 3) 전제 및 확인 항목

- 사용자 가정: 학습/추론 시 instance가 256으로 capped.
- 코드 확인상 RF-DETR segmentation 기본 query/select는 size별 100/200/300으로 구성됨.
- 따라서 `v9` 시작 전, 실제 cap이 어디에서 걸리는지 로그로 확정한다.
  - 후보: `num_queries`, `num_select`, SAHI postprocess, 후처리 threshold 조합
- 계획상 목표는 동일: 타일링으로 타일당 객체 수를 cap 이하로 분산.

## 4) v9 핵심 전략

1. 오프라인 타일 데이터셋 생성(권장)
2. `tile_size = train resolution`(예: M=432, XL=624, 2XL=768)
3. 타일 경계 기준으로 segmentation polygon clip 후 유효 조각만 유지
4. 타일당 annotation 분포를 관리해 `p95 <= 180`(권장), `max <= 256`(하드 목표)
5. 추론(`inference.py` SAHI)도 tile/slice 설정을 학습과 맞춰 도메인 갭 감소

## 5) 데이터 생성 설계 (train/valid 동일 규칙, split 분리 유지)

### 입력/출력

- 입력: `/home/mbd1234/data/Optiresolve_result_total_20260223/total_data_refine_1024/{train,valid}`
- 출력(예시): `/home/mbd1234/data/Optiresolve_result_total_20260223/total_data_refine_1024_tiled_r{resolution}_ov{overlap}/{train,valid}`

### 타일 규칙

- `tile_size`: 모델 resolution과 동일
- `overlap`: 기본 0.15~0.20 (초기값 0.20 권장)
- `stride = tile_size * (1 - overlap)`
- 마지막 경계 타일 포함(우/하단 누락 금지)
- 파일명 예: `origStem__x{left}_y{top}_s{tile}.jpg`

### annotation clip 규칙

- 타일과 교차하는 annotation만 처리
- polygon을 타일 경계로 clip 후 타일 로컬 좌표로 변환
- 필터 기준(초기안):
  - clipped area >= 16 px
  - visible_ratio(clipped/original) >= 0.10
  - bbox width/height >= 2 px
- `bbox`, `area`, `segmentation` 재계산
- `category_id`, `iscrowd` 유지

### 빈 타일 처리

- 객체 0개 타일은 전부 저장하지 않고 비율 제한(예: 양성 타일 대비 최대 20%)
- 목적: 배경 학습은 유지하되 데이터 폭증 방지

## 6) trainV9.py 변경 계획 (trainV8 기반)

- `trainV8.py`의 구조는 유지:
  - resolution 검증, 모델 생성, `call_train` 필터링, `run_meta.json`, wandb 처리
- `trainV9.py` 추가/변경 항목:
  - 타일 데이터셋 사용 옵션:
    - `--tile-enable`
    - `--tile-size` (기본: `resolution`)
    - `--tile-overlap` (기본: `0.20`)
    - `--tile-min-area`
    - `--tile-min-visible-ratio`
    - `--tile-output-root`
    - `--tile-rebuild` (기존 타일 강제 재생성)
  - 실행 모드:
    - `raw`: 기존 원본 데이터 그대로 학습
    - `tile`: 타일 데이터셋 생성 후 학습
  - `run_meta.json`에 타일 파라미터/분포 통계 기록
  - `--sanity-check` 확장:
    - 타일당 annotation 분포 요약(mean/median/p95/max)
    - cap 초과 타일 경고 출력

## 7) 추론(inference.py) 정렬 계획

- 학습과 추론의 단위 스케일을 맞춘다.
- 권장:
  - `--slice-size` = 학습 `tile_size`
  - `--overlap` = 학습 `tile-overlap`과 동일 또는 근접값
- 비교 실험:
  - A: 기존 추론 설정
  - B: 학습-정렬 슬라이싱 설정
  - 지표: annotation recall, 과분할/미검출 패턴, 이미지당 검출 수

## 8) 실험 계획 (단계별)

1. Cap 원인 확인
  - 현재 모델/후처리 로그로 실제 상한(예: 256) 확인
2. 타일 데이터셋 생성 스크립트 작성
  - train/valid 타일 생성 + COCO 재구성 + 통계 리포트
3. 스모크 학습 (10~20 epoch)
  - trainV8 + tiled dataset로 빠른 안정성 확인
4. trainV9 통합
  - 타일 생성/재사용을 CLI에서 제어
5. 본학습 + 비교평가
  - raw vs tiled 성능/속도/메모리 비교

## 9) 성공 기준

- 타일 데이터셋에서 `ann_per_tile p95`가 목표치(권장 180 이하)에 근접
- valid segmentation 성능(특히 밀집 이미지군 recall) 개선
- 학습 안정성 향상(OOM/불안정 step 감소)
- 추론 결과에서 과밀 이미지 누락 감소

## 10) 바로 다음 실행 액션

1. `trainV9.py`에 들어갈 타일링 로직을 별도 유틸 스크립트로 먼저 분리 구현
2. 타일 생성 후 분포 리포트(`ann_per_tile`)를 저장
3. 같은 하이퍼파라미터로 raw vs tile 20epoch 비교 실험 수행

