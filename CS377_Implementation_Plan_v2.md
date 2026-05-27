# 구현 계획서 — 재료 핸디캡 체스에서 집중 vs 분산 제거 (상세)

**과목:** CS377 강화학습 · **Team 1**
**관련 문서:** *프로젝트 정리(Project Summary)* · 본 문서는 그 엔지니어링 빌드 가이드.

---

## 0. 설계가 단순해진 지점 (먼저 인식)

이번 설계는 **규칙 변종이 아니라 "표준 체스를 합법적 재료-down 위치에서 시작"**하는 것이다. 그래서 이전 free-setup 버전 대비:

- **표준 규칙 그대로** → 캐슬링·앙파상 비활성화 불필요. **python-chess를 거의 off-the-shelf로** 사용. 커스터마이즈는 *시작 위치(어떤 기물을 제거)* 뿐.
- **배치 자유도 0** → 킹 랜덤화·setup 주입·배치 교란 로직 전부 사라짐.
- **명시적 conditioning plane이 (거의) 불필요** → 제거된 기물은 보드에 *없음*으로 그대로 드러나므로, 보드 입력만으로 재료 구성이 완전히 표현된다. 즉 "조건화"의 본질은 *self-play를 핸디캡 위치들에서 초기화*하는 것이고, net은 평소 보드 입력으로 자동 조건화된다. (원한다면 제거-패턴 feature plane을 추가할 수 있으나 redundant.) **통제 비교 기여는 "하나의 net이 모든 조건을 둔다"는 점에서 그대로 성립.**
- **기존 체스 지식으로 sanity check 가능** → 퀸-down ≈ 패배 같은 알려진 평가로 구현 검증.

---

## 1. 두 트랙 구조 (하이브리드)

| 트랙 | 목적 | 산출물 | 리스크 |
|---|---|---|---|
| **A. 직접 conditional AZ** | RL 수업 요구 충족(직접 구현·학습) | 학습 가능 증명 + 자기 실력 수준의 결과 | 높음(엔지니어링·throughput) |
| **B. lc0 강한 레퍼런스** | 신뢰성 있는 *메인* 결과(강한 플레이) | 조건별 다수 게임의 W/D/L | 낮음(사전학습 net 재사용) |
| **C. 분석** | 두 트랙 게임 로그 소비 | 가산성 검정·유효 기물 점수·흑백 비대칭 | 중간 |

**하이브리드 논리:** 메인 과학 결과는 **B(lc0)**가 강한 플레이로 책임지고, **A(직접 AZ)**는 RL 역량 시연 + 자기 실력 수준에서 B의 추세를 corroborate. 이렇게 나누면 A가 superhuman일 필요가 없어져 리스크가 크게 준다. **B를 먼저(Phase 1) 돌려 과학을 일찍 de-risk**하고, A는 병렬로 진행.

---

## 2. 공통 인프라

### 2.1 핸디캡 위치 생성기 `handicap.py`
- 제거 패턴 config(점수 9 고정). 각 패턴마다 **제거할 구체 칸**을 명시(잔여 설계 결정): 어느 폰(중앙/가장자리), 라이트/다크 비숍, 어느 나이트/룩. 한 번 정하고 고정·정당화.
- python-chess `Board()`에서 해당 칸 기물 제거 → **FEN** 출력. 핸디캡을 백/흑 양쪽으로 생성(color balance).
- 출력: `{pattern_id, side, fen, material_diff_vector(ΔQ,ΔR,ΔB,ΔN,ΔP)}`.

### 2.2 게임 로그 스키마
한 게임당 기록: `pattern_id, handicap_side, color_of_handicap, result(W/D/L from handicap side), ply, start_fen, (선택)termination_reason, engine/agent, nodes`. 분석은 전적으로 이 로그만 소비.

---

## 3. Track B — lc0 파이프라인 (가장 빠른 결과 경로)

### 3.1 구성 `lc0_runner.py`
- python-chess `chess.engine.SimpleEngine`(UCI)로 lc0 구동. 각 게임은 `position fen <핸디캡 FEN>`에서 시작.
- 두 인스턴스(또는 한 net 자가대국). **고정 nodes/sims**로 강도 통제, 오프닝북 off.
- **오프닝 다양성**: 첫 ~10–16 ply를 temperature로 샘플링(Tomašev 방식) → 같은 조건에서 독립 표본 확보. 이후 결정적.
- color-balanced: 각 패턴을 핸디캡-백 / 핸디캡-흑 균등.

### 3.2 (선택) Rating 비대칭
강한 쪽 = 많은 nodes, 약한 쪽 = 적은 nodes로 두 인스턴스 강도를 다르게 → "어떤 제거가 점수를 0.5에 가장 가깝게(=가장 공정) 만드나" 측정.

### 3.3 검증
- 시작 위치들이 lc0에 *out-of-distribution*(1수째부터 9점 down)일 수 있으므로, **샘플 위치 eval이 합리적인지**(퀸-down이 크게 불리로 평가되는지 등) 먼저 확인.

---

## 4. Track A — 직접 conditional AlphaZero (RL 구현)

이전 버전의 좋은 골격(게임 비종속 코어 → 사소한 게임으로 먼저 증명)은 유지하되, env를 표준 체스로 단순화.

### 4.1 아키텍처 — 게임 비종속 코어
`Game` ABC로 코어(MCTS·net·self-play·trainer)와 게임을 분리. **tic-tac-toe로 코어를 먼저 증명**한 뒤 표준 체스를 끼운다(조용한 MCTS 버그를 체스 복잡도와 분리).

### 4.2 모듈

| 모듈 | 책임 | 핵심 함정 | 테스트 |
|---|---|---|---|
| `game/base.py` | `Game` ABC | 인터페이스 비대화 | 타입 |
| `game/tictactoe.py` | 검증용 게임 | — | 합법 종료 |
| `game/chess_std.py` | python-chess 래퍼(표준 규칙) + **커스텀 init(핸디캡 FEN)** | 표준이라 거의 그대로 | **perft**(표준 알려진 값) |
| `encoding.py` | 보드 plane + 8×8×73 action + mask | mask는 softmax 전; 왕복 정확성 | encode/decode 라운드트립 |
| `net.py` | ResNet, policy+value (PyTorch) | value `tanh` 범위 | 1-batch overfit |
| `mcts.py` | PUCT·expand·backup·root Dirichlet·temperature | **backup 부호(canonical form)** | TTT 전술 해결 |
| `selfplay.py` | **핸디캡 위치들에서 self-play 초기화** · 배치 eval · replay buffer | 시작분포 = 조건 분포 | throughput 측정 |
| `trainer.py` | loss(policy CE + value MSE + L2)·optimizer·AMP | target detach·LR | TTT loss↓ |
| `arena.py` | 학습 net으로 조건별 대국, color-balanced | eval시 noise off·τ=0 | 시드 결정성 |

### 4.3 State / Action / Reward (요약 — 상세는 별도 인코딩 명세)
- **State**: 표준 보드 plane(12 기물 + 반복 2 + side/진행도). **conditioning plane 불필요**(보드가 재료를 표현). canonical form 권장.
- **Action**: AZ **8×8×73** 그대로. python-chess가 합법수 생성, mask가 불법 0.
- **Reward**: 종단 z ∈ {+1 승/체크메이트, 0 무, −1 패}. shaped reward 없음. 표준 종료 규칙(체크메이트·스테일메이트·삼수동형·50수).

### 4.4 "조건화"의 실제 구현
명시적 plane 대신 **self-play 게임을 핸디캡 위치 분포에서 시작**(우선 패턴들 + 양 color를 샘플링). net은 보드 입력으로 자동 조건화되고, **하나의 net이 모든 조건을 학습** → 통제 비교 기여 성립. (제거-패턴 one-hot plane 추가는 선택.)

### 4.5 검증 게이트
표준 체스라 *알려진 지식으로 검증 가능*:
- 학습이 일어나는가: loss↓, policy entropy↓, 약한 baseline(저-node lc0 또는 depth-제한 탐색) 상대 승률↑.
- **sanity**: 학습 net의 핸디캡 위치 평가가 lc0/체스 통념과 *대략* 일치(퀸-down 불리 등). 어긋나면 인코딩·MCTS 의심.

---

## 5. Track C — 분석 (과학적 payoff)

### 5.1 승률 + 가산성 검정 (Main RQ)
- 패턴별 핸디캡 측 점수(승+0.5무)를 **Dirichlet-multinomial**(W/D/L 3-way)로 추정, 95% CrI. color-balanced로 선수 이점 상쇄.
- **가산성**: 패턴 간 점수가 통계적으로 동일한가. 동일→ "9점은 어떻게 빼도 같음". 다름→ 집중도가 유효 가치를 바꿈. 패턴 이질성을 정직히 다루려면 hierarchical 모델 권장.

### 5.2 유효 기물 점수 회귀 (Sub-RQ 2) — 핵심
- 모델: `logit P(핸디캡 측 승) = β0 + βQ·ΔQ + βR·ΔR + βB·ΔB + βN·ΔN + βP·ΔP + βcolor·color`
  여기서 Δ = 핸디캡으로 *제거된* 각 기물 수. β_piece = **작동상 유효 가치**(음수). 표준 {9,5,3,3,1}과 비교.
- ⚠️ **식별성 주의**: 모든 패턴이 9점 고정이라 *총* 점수차가 상수 → 디자인 행렬 rank가 제한됨. *상대적* 기여는 추정되나 절대값은 anchor가 필요. **대응: 패턴을 많이 쓸수록(7개 전체) 구성 변동↑ → 식별 개선.** 회귀용으로는 넓은 패턴 집합, head-to-head용으로는 패턴당 다수 게임 — 이 트레이드오프를 의식해 표본 배분.

### 5.3 흑백 비대칭 (Sub-RQ 3)
- 각 패턴을 핸디캡-백 / 핸디캡-흑 양쪽으로 → β_color 및 색별 유효 가치 차이 측정(선수 이점과 재료의 상호작용).

---

## 6. 마일스톤 (게이트 포함) — 통합 일정

각 게이트 통과 전엔 다음으로 넘어가지 않는다. **B(과학)를 앞에, A(RL)는 병렬.**

| M | 산출 | **통과 테스트** |
|---|---|---|
| **M0** | 레포·config·`handicap.py`·로그·분석 스텁·`Game` ABC·TicTacToe | 핸디캡 FEN 생성·검증; TTT 합법 종료 |
| **M1** | **Track B lc0 파이프라인 → 예비 결과** | ≥1 패턴 W/D/L 산출; 샘플 eval이 체스 통념과 부합 |
| **M2** | 직접 AZ 코어, TTT 완벽 학습 (RL 코어 증명) | TTT 완벽 플레이(무패) |
| **M3** | 배치 self-play throughput | games/sec 측정·학습량 계획 |
| **M4** | `chess_std.py` env + encoding | **perft**(표준 값) 일치; encode/decode 라운드트립 |
| **M5** | 단일 핸디캡 조건 self-play 학습 | loss↓, 약 baseline 격파, eval이 lc0와 대략 일치 |
| **M6** | 전 우선 패턴 단일 net 학습 | 모든 우선 패턴 competent; (선택) B 추세 corroborate |
| **M7** | 두 트랙 게임 생성 + Track C 분석 | posterior·CrI; 가산성 판정; 유효 가치·흑백 비대칭 |

M1이 과학을 일찍 확보하므로, A가 늦어져도 결과 자체는 위험하지 않다.

---

## 7. 테스트 전략

- **마스터 통합 테스트(M2)**: "TTT 완벽 학습" — 코어 회귀 가드로 상시 유지.
- **perft(M4)**: 표준 체스 알려진 노드 수와 대조(표준이라 reference 풍부).
- **encode/decode 라운드트립(M4)**: 모든 합법 수 왕복; mask 합 = 합법수.
- **1-batch overfit(M2,M5)**: net이 한 배치를 ~0 loss로 외우는가(죽은 gradient·detach 버그).
- **MCTS uniform-prior sanity**: 균등 net + 다수 sims → 해결된 TTT 위치에서 좋은 수에 방문 집중.
- **분석 합성 데이터 검증(M0)**: 알려진 승률·기물값을 심은 가짜 로그에서 추정기가 복원하는가.
- **결정성**: 고정 시드 → 재현.

---

## 8. 컴퓨트 & throughput

- **Track B(lc0)**: 적당한 net·고정 nodes면 GPU에서 빠름 — 패턴×수천 게임 현실적. 메인 결과의 부담을 짊어짐.
- **Track A(직접)**: 작은 net 유지. 배치 leaf 평가(여러 게임 병렬)로 GPU 비우지 않기. A는 superhuman 불요 → **규모를 줄여도 됨**(학습 증명 + 추세 확인이 목표). throughput은 M3에서 *측정* 후 학습량 역산.
- 컴퓨트 부족 시 레버: A의 패턴 수↓(우선 3개) → 게임 수↓ → sims↓. **A를 줄이되 B는 유지**.

---

## 9. 레포 구조

```
handichess/
  config/            # 제거 패턴(칸 지정)·net·mcts·train·arena·lc0 설정
  common/
    handicap.py      # 패턴→FEN 생성기
    gamelog.py       # 로그 스키마/IO
  track_b/
    lc0_runner.py    # UCI 구동, 다양성, color balance
  track_a/
    game/ base.py tictactoe.py chess_std.py
    encoding.py net.py mcts.py selfplay.py trainer.py arena.py
    baseline.py      # 약 baseline(저-node lc0 등)
  analysis/
    winrate.py       # Dirichlet-multinomial
    piece_values.py  # 로지스틱 회귀
    color_asym.py
  tests/             # perft, roundtrip, overfit, mcts-sanity, ttt-learns, synth-analysis
  scripts/           # gen_positions, run_lc0, train, run_arena, analyze
  runs/              # 체크포인트·로그(gitignore)
```

## 10. 역할 분담 (3인)

`Game` 인터페이스와 로그 스키마가 계약. M2 이후 병렬화.
- **A — Track B + 공통 인프라**: `lc0_runner`, `handicap.py`, `gamelog`. M1 책임.
- **B — Track A net/env/학습**: `chess_std.py`, `net.py`, `trainer.py`, encoding. M4·M5의 net측.
- **C — Track A 탐색/self-play + 분석**: `mcts.py`, `selfplay.py`(throughput), `arena.py`, `analysis/`. M2 코어·M3·M7.
- 공동: M2(코어 증명)·M6(학습 run).

## 11. 리스크 & 완화

| 리스크 | 가능성 | 완화 |
|---|---|---|
| 직접 AZ가 약하거나 느림 | 높음 | **B(lc0)가 메인 결과를 짊어짐**; A는 RL 시연·추세 확인용 |
| lc0가 1수째 imbalance에 OOD | 중 | 샘플 eval 사전 검증; 충분한 nodes |
| 회귀 식별성(9점 고정 제약) | 중 | 패턴 수↑로 구성 변동 확보; 상대값 추정·anchor |
| RL 수업 요구(직접 구현 필요) | — | Track A가 구현 콘텐츠 보장 |
| 폰-heavy 제거의 구조 교란 | 중 | 우선은 폰 손상 적은 패턴; 폰-heavy는 별도 "구조" 축 |
| MCTS 조용한 버그 | 중 | M2 TTT 게이트가 사전 차단 |

## 12. 확정 의존성 (Project Summary §8과 연결)

진행 전 잠글 것: **(1) 우선 패턴 집합**, **(2) 제거할 구체 칸**, **(3) A의 규모·B와의 경계**, **(4) rating 비대칭 여부**, **(5) 명시적 조건화 plane 사용 여부(기본=불필요)**.

---

*다음 단계 후보: (a) 인코딩 명세(보드 plane 인덱스 표 + 8×8×73 매핑 의사코드), (b) `handicap.py`의 패턴-칸 config 초안, (c) `lc0_runner` 게임 루프 의사코드, (d) `piece_values.py` 회귀 모델 구체화.*
