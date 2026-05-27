# AlphaZero Chess (Design B) 클러스터 실행 가이드

이 문서는 4x GTX 5050 클러스터 환경에서 "집중(Q) vs 분산(NoQ)" 매치업(Design B) 실험을 수행하기 위한 단계별 실행 가이드(Runbook)입니다.

## 0. 환경 준비

클러스터 노드에 접속한 후 최상위 디렉토리에서 의존성 및 환경을 활성화합니다.
*(이 프로젝트는 Python 3.9 이상 및 PyTorch 환경을 필요로 합니다)*

```bash
# 가상환경 활성화 (예시)
source .venv/bin/activate

# 경로(PYTHONPATH) 인식 (필수)
export PYTHONPATH=$(pwd)
```

**사전 요구사항:** LC0 (Leela Chess Zero) 엔진 바이너리가 `PATH`에 존재하거나, 프로젝트 루트에 `lc0` 실행 파일이 위치해야 평가(Track B)를 정상적으로 수행할 수 있습니다.

---

## 단계 1: 폰-Heavy 묶음 파일럿 테스트 (Floor Pilot)

가장 불안정한 2개의 폰-다수 패턴(`bishop_6pawns`, `knight_6pawns`)이 변별 구간(0.3 ~ 0.7) 내에 들어오는지, 아니면 한쪽으로 극단적으로 쏠리는지 확인합니다.

```bash
python scripts/run_floor_pilot.py
```

- **예상 소요 시간:** 20판 (패턴당 10판) × 100노드 기준 약 1~2분 소요
- **확인 사항:** 터미널 출력(Log)에서 `[PILOT RESULT]` 라인을 찾아 Q 진영의 승률이 `0.3 ~ 0.7` 사이에 있는지 확인합니다. 만약 0.9 이상이거나 0.1 이하라면 메인 분석 시 이 두 패턴은 분리해서 해석해야 합니다.

---

## 단계 2: Track B 메인 평가 (LC0 기준점 측정)

이 프로젝트의 메인 가설(집중이 분산을 능가하는가?)을 증명하기 위한 **가장 핵심적인 데이터 추출 단계**입니다. 통제된 환경(고정 시드, 교차 진영)에서 평가를 진행합니다.

```bash
# 전체 7개 패턴에 대해 LC0(800 노드)를 이용해 1000판씩 평가 진행
python scripts/run_lc0.py \
    --engine lc0 \
    --nodes 800 \
    --games 1000 \
    --output runs/lc0_baseline.jsonl
```

- **옵션 설명:**
  - `--nodes 800`: LC0가 매 턴마다 탐색하는 노드 수(논문 표준 수준 강도).
  - `--games 1000`: 각 패턴마다 1000판씩 수행 (백 500판, 흑 500판). 신뢰 구간 95% 이상을 확보할 수 있는 통계적 유의미한 수치입니다.
- **결과물:** `runs/lc0_baseline.jsonl` 파일에 각 패턴별 게임 결과와 승률(Q측 관점)이 누적 기록됩니다.
- *(추후 `scripts/analysis/*.py` 분석 스크립트로 이 JSONL 파일을 시각화할 수 있습니다.)*

---

## 단계 3: Track A 메인 훈련 (AlphaZero Self-Play)

통제 도구(LC0)의 결과와 AlphaZero가 스스로 터득한 기물 가치의 결론이 일치하는지(Corroboration) 검증하기 위해 **단일 에이전트**를 7가지 모든 패턴에서 훈련시킵니다. 4개의 GPU를 풀 가동합니다.

```bash
python scripts/train.py \
    --game chess \
    --iterations 150 \
    --games-per-iter 1024 \
    --simulations 800 \
    --checkpoint-dir runs/checkpoints
```

- **멀티 GPU 최적화:** 코드가 내부적으로 `torch.nn.DataParallel` 및 Batched MCTS를 사용하여 가용한 모든 GPU(4x 5050)를 자동 활용하도록 설계되어 있습니다.
- **다양성 훈련:** `train.py`는 게임을 시작할 때마다 7개의 패턴 중 하나를 **무작위(Random)**로 골라 Self-Play를 수행하므로, 하나의 모델이 모든 상황에 대처하는 범용 체스 원리를 깨우치게 됩니다.

---

## 단계 4: Track A 모델 평가 (Arena)

훈련된 AlphaZero 에이전트(Track A)의 성과를 최종 측정합니다. 가장 잘 학습된 체크포인트(예: `checkpoint_0050.pt`)를 가져와 Baseline(Random, Greedy, 또는 Weak MCTS)과 각 패턴별로 승률을 측정합니다.

```bash
# 특정 패턴(예: rook_bishop_pawn)에 대해 1000판 평가
python scripts/run_arena.py \
    --game chess \
    --checkpoint runs/checkpoints/checkpoint_0150.pt \
    --baseline greedy \
    --games 1000 \
    --pattern rook_bishop_pawn
```

> **팁:** Bash 루프를 사용하여 7개 패턴 모두에 대해 자동 평가를 돌릴 수 있습니다.
```bash
for pat in rook_bishop_pawn rook_knight_pawn bishop_bishop_knight rook_4pawns bishop_knight_3pawns bishop_6pawns knight_6pawns; do
    echo "Evaluating pattern: $pat"
    python scripts/run_arena.py --game chess --checkpoint runs/checkpoints/checkpoint_0150.pt --baseline greedy --games 1000 --pattern $pat
done
```
