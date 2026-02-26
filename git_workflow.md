# Git Workflow (cyh5757 fork <-> roboflow upstream)

## 0) 브랜치 역할
- `develop` -> `roboflow/rf-detr`의 `develop` 미러용 (직접 작업 X)
- `feature/my-work` -> 내 작업 브랜치 (여기서만 수정/커밋)

## 1) 지금 내가 어느 브랜치인지 확인
```bash
git branch --show-current
git status -sb
```

## 2) 평소 작업 순서 (항상 이 순서)
`feature/my-work에서 코드 수정 -> git add -> git commit -> git push origin feature/my-work`

예시:
```bash
git switch feature/my-work
git add .
git commit -m "feat: ..."
git push origin feature/my-work
```

## 3) roboflow 최신 업데이트 받는 순서 (충돌 최소화)
`(작업 중이면 먼저 commit 또는 stash) -> fetch upstream -> develop 동기화 -> origin/develop 반영 -> feature 재배치(rebase) -> push`

명령어:
```bash
git fetch upstream

git switch develop
git reset --hard upstream/develop
git push origin develop --force-with-lease

git switch feature/my-work
git rebase develop
git push --force-with-lease origin feature/my-work
```

## 4) rebase 중 conflict 나면
`충돌 파일 수정 -> git add <충돌파일> -> git rebase --continue -> 끝나면 push --force-with-lease`

명령어:
```bash
git status
# 충돌 해결 후
git add <file1> <file2>
git rebase --continue

# 모두 끝난 뒤
git push --force-with-lease origin feature/my-work
```

## 5) 절대 하지 말 것
- `develop`에서 직접 코드 수정/커밋
- rebase 도중 상태에서 바로 push 시도
- `git push --force` 남발 (`--force-with-lease` 사용)

## 6) PR 보낼 때
`origin/feature/my-work -> roboflow/develop` 로 PR 생성

## 7) 빠른 체크리스트
- 작업 시작 전: `git switch feature/my-work`
- 업스트림 반영 전: 미커밋 변경 없는지 확인 (`git status`)
- 반영 후: `git status -sb` / `git log --oneline --decorate -n 5`
