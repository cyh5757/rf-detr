# trainV10.py CLI 사용법

## 핵심 포인트
- `trainV10.py`는 타일 데이터셋 생성(`--tile-*`) + 타일 오프라인 증강(`--aug-*`)을 지원합니다.
- 추가로 RF-DETR 기본 `aug_config`도 학습 시점에 전달할 수 있습니다.
  - `--train-aug-preset` (내장 프리셋)
  - `--train-aug-config-json` (커스텀 JSON)
- 두 증강은 동시에 사용할 수 있습니다.
  - `--aug-*`: 타일 이미지를 디스크에 증강본으로 추가 저장
  - `--train-aug-*`: `model.train(aug_config=...)`로 온라인 증강

## 빠른 시작
```bash
cd /home/mbd1234/rf-detr
python trainV10.py \
  --size 2XL \
  --resolution 768 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 0 \
  --amp \
  --gradient-checkpointing \
  --tile-enable
```

## augmentation 사용 예시

### 1) RF-DETR 기본 augmentation 사용 (기본값)
`--train-aug-preset default`가 기본입니다.  
이 경우 `aug_config`를 명시 전달하지 않고, RF-DETR 기본 `AUG_CONFIG`가 적용됩니다.

```bash
python /home/mbd1234/rf-detr/trainV10.py \
  --size 2XL \
  --resolution 768 \
  --tile-enable
```

### 2) 내장 프리셋 사용
```bash
python /home/mbd1234/rf-detr/trainV10.py \
  --size 2XL \
  --resolution 768 \
  --tile-enable \
  --train-aug-preset aggressive
```

지원 프리셋:
- `default`: RF-DETR 기본 `AUG_CONFIG` 사용
- `none`: 학습 augmentation 비활성화 (`aug_config={}`)
- `conservative`
- `aggressive`
- `aerial`
- `industrial`

### 3) 커스텀 JSON 사용
`--train-aug-config-json`을 주면 `--train-aug-preset`보다 우선합니다.

```bash
python /home/mbd1234/rf-detr/trainV10.py \
  --size 2XL \
  --resolution 768 \
  --tile-enable \
  --train-aug-config-json /home/mbd1234/rf-detr/aug_custom.json
```

`aug_custom.json` 예시:
```json
{
  "HorizontalFlip": { "p": 0.5 },
  "Rotate": { "limit": 15, "p": 0.3 },
  "GaussianBlur": { "p": 0.2 }
}
```

## 타일 오프라인 증강(`--aug-*`)과 함께 쓰기
아래 예시는 두 증강을 동시에 사용합니다.

```bash
python /home/mbd1234/rf-detr/trainV10.py \
  --size 2XL \
  --resolution 768 \
  --tile-enable \
  --aug-enable \
  --aug-policy conservative \
  --aug-train-copies 1 \
  --train-aug-preset conservative
```

## 주요 인자 (augmentation 관련)
- `--aug-enable`: 타일 오프라인 증강 활성화
- `--aug-policy {conservative,aggressive}`: 타일 오프라인 증강 정책
- `--aug-train-copies`: train 타일당 추가 생성할 오프라인 증강본 수
- `--aug-verify-count`: 오프라인 증강 preview 저장 개수
- `--aug-verify-dir`: 오프라인 증강 preview 출력 폴더
- `--aug-verify-only`: 타일/오프라인 증강 생성 후 학습 없이 종료
- `--train-aug-preset {default,none,conservative,aggressive,aerial,industrial}`: 학습 `aug_config` 프리셋
- `--train-aug-config-json`: 학습 `aug_config` 커스텀 JSON 파일 경로 (preset보다 우선)

## 출력/기록
- 학습 출력 폴더의 `run_meta.json`에 아래가 기록됩니다.
  - 타일 오프라인 증강 설정(`tile_aug_*`)
  - 학습 `aug_config` 설정(`train_aug_*`)

