# trainV8.py CLI 사용법

## 핵심 변경점
- `--resolution`이 태그용이 아니라 **실제 학습 입력 해상도**로 적용됩니다.
- `--amp` / `--no-amp` 옵션으로 mixed precision을 명시적으로 제어할 수 있습니다.
- `--gradient-checkpointing` 옵션으로 activation 메모리를 줄일 수 있습니다(속도 저하 가능).

## 기본 실행
```bash
python /home/mbd1234/rf-detr/trainV8.py \
  --data-root /home/mbd1234/data/Optiresolve_result_total_20260223/total_data_refine_1024 \
  --size M \
  --resolution 480 \
  --epochs 150 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --num-workers 0 \
  --amp \
  --gradient-checkpointing
```

## OOM 완화 실행 예시
```bash
python /home/mbd1234/rf-detr/trainV8.py \
  --size 2XL \
  --resolution 768 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --num-workers 0 \
  --amp \
  --gradient-checkpointing
```

## 주요 인자
- `--data-root`: COCO 형식 데이터셋 루트 (`train/_annotations.coco.json`, `valid/_annotations.coco.json`)
- `--outputs-root`: 출력 상위 폴더
- `--size {N,S,M,L,XL,2XL}`: 모델 크기
- `--resolution`: 학습 해상도 (미지정 시 size 기본값 사용)
- `--epochs`: 총 학습 epoch
- `--batch-size`: mini-batch 크기
- `--grad-accum-steps`: gradient accumulation step
- `--num-workers`: DataLoader worker 수
- `--pad-to-square` / `--no-pad-to-square`: 정사각 패딩 여부
- `--lr`: learning rate
- `--amp` / `--no-amp`: mixed precision 사용 여부 (기본값: `--amp`)
- `--gradient-checkpointing`: gradient checkpointing 사용
- `--num-classes`: 클래스 수
- `--class-names`: 클래스 이름 목록
- `--early-stopping`: early stopping 사용
- `--resume`: 체크포인트에서 resume
- `--sanity-check`: segmentation polygon 샘플 검사 실행
- `--wandb-project`, `--wandb-run-name`, `--wandb-disable`: wandb 제어

## resolution 규칙
`trainV8.py`는 모델 구조 제약에 맞는 해상도만 허용합니다.

- `N`: `12`의 배수
- `S/M/L/XL/2XL`: `24`의 배수 (`patch_size=12`, `num_windows=2`)

예:
- `M`에서 유효: `432`, `456`, `480`, `504`, ...
- `M`에서 무효: `450` (24의 배수가 아님)

## 출력 파일
`--outputs-root/<dataset_tag>/<timestamp__...>` 아래에 저장됩니다.

- `checkpoint_best_total.pth`
- `checkpoint_best_ema.pth`
- `run_meta.json`

## 자주 발생하는 이슈
- `DataLoader worker crashed`
  - 스크립트가 자동으로 `num_workers=0`으로 1회 재시도합니다.
- `resolution ... 배수여야 함`
  - 위 배수 규칙에 맞는 값으로 수정하세요.
- `wandb 사용 불가`
  - `pip install wandb` 또는 프로젝트 내 `wandb.py` 파일 충돌 여부 확인

## 빠른 시작 커맨드
```bash
cd /home/mbd1234/rf-detr
python trainV8.py --size M --resolution 480 --epochs 150 --batch-size 2 --grad-accum-steps 2 --amp --gradient-checkpointing
```
