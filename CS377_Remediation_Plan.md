# 수정 계획서 — HandiChess AlphaZero 코드베이스

**대상 레포:** `CS377_Team1_...via-AlphaZero` · **작성 근거:** 코드 리뷰(2026-05)
**원칙:** 모든 수정은 **Red → Fix → Green** — *버그를 재현하는 실패 테스트를 먼저 작성*하고, 고친 뒤 통과를 확인한다. 지금 테스트 스위트가 못 잡는 사각지대(흑 차례 체스, end-to-end 학습)를 메우는 것이 절반의 목표다.

---

## 0. 진단 요약

AlphaZero 파이프라인(Game ABC ↔ MCTS ↔ Net ↔ SelfPlay ↔ Trainer)은 **구조적으로 완성**되어 있고 알고리즘도 정통 AZ다. 그러나 **체스에 한해 두 개의 치명적 버그**가 있어 현재 상태로 학습하면 결과가 무효다. 더 위험한 것은 **테스트가 모두 통과하면서 이 버그들을 못 잡는다**는 점(흑 차례 체스·end-to-end 학습 경로를 건드리는 테스트가 없음). TicTacToe 골격은 동작 가능성이 높으나 실제 학습 성공은 미검증 상태(체크포인트·로그 없음).

---

## 1. 수정 우선순위

| ID | 문제 | 심각도 | 위치 |
|----|------|--------|------|
| **P0-1** | canonical/action 좌표계 불일치 — 흑 차례 정책·mask·학습타깃 오염 | 🔴 차단 | `mcts.py::_expand`, `game/chess_std.py::get_canonical_form`, `selfplay.py::play_game` |
| **P0-2** | 상태를 bare FEN으로 저장 → 히스토리 손실 → 반복/삼수동형 감지 사망, 반복 plane 죽음 | 🔴 차단 | `game/chess_std.py`(`_fen_to_array` 등), `encoding.py`(repetition planes) |
| **P1-1** | 흑 underpromotion decode가 잘못된 칸 생성 → 불법수 assert 크래시 가능 | 🟠 높음 | `encoding.py::_decode_move_plane`, `decode_action` |
| **P1-2** | 테스트 거짓 안심 — 흑 차례·승급·end-to-end 미검증 | 🟠 높음 | `tests/` |
| **P2-1** | throughput — 비배치 NN 평가 + 매 노드 FEN 재파싱 + `is_repetition` 스캔 | 🟡 중간 | `mcts.py`, `net.py::predict`, `game/chess_std.py` |
| **P2-2** | 기타(net 크기, value dropout, sqrt(0) prior, dead code) + 미확인(분석 식별성, 바깥 루프) | ⚪ 낮음/확인 | 다수 |

---

## 2. 개별 항목 — 문제 / 방향성 / 검증

### 🔴 P0-1. canonical/action 좌표계 불일치

**문제.** `get_canonical_form(state, -1)`은 `board.mirror()`로 보드를 상하 반전+색 교환한다. `mcts._expand`는 이 *미러 보드*를 인코딩해 정책을 얻지만, `valid_moves`·자식 생성·`prior=policy_probs[action]`은 *실제(un-mirror) 좌표*를 쓴다. → 흑 차례엔 정책이 미러 좌표, mask/action이 실제 좌표라 어긋난다. 게다가 `selfplay.play_game`이 `(미러 인코딩, 실제-좌표 π)`로 학습 예제를 저장해 **학습 데이터까지 오염**된다. 백(+1)은 canonical==actual이라 무사, 흑만 깨진다.

**방향성 (둘 중 하나, 전 경로 일관 적용).**
- **(A · 권장, 최소 변경)** 체스에서 공간 canonical을 포기: `get_canonical_form`을 `return board.copy()`(no-op)로. `encode_board`가 이미 `is_own = piece.color == turn`으로 소유권을 관점화하고 color plane(14)도 있으므로, mirror만 안 하면 정책·mask·자식·학습 π가 전부 실제 좌표로 일관된다. 비용: net이 흑/백 두 방향을 모두 학습(약간의 샘플 비효율, 정확성엔 문제 없음).
- **(B · 정석, 더 많은 작업)** mirror 유지하되 ① mask를 canonical 보드에서 생성, ② 자식 prior를 읽을 때 *실제 move를 미러→`encode_action`*한 canonical 인덱스로 읽기, ③ 학습 π도 canonical 좌표로 저장. 단일 방향 학습이라 샘플 효율은 좋지만 버그 위험↑.

> 과제 일정상 **(A)** 권장. 무엇을 택하든 **`mcts._expand`(탐색)와 `selfplay.play_game`(학습 타깃 저장) 양쪽에 동일하게** 적용할 것.

**검증.**
1. *프레임 일관성 단위 테스트*(지금 실패 → 수정 후 통과): 흑 차례 위치에서
   `get_legal_move_mask(canonical_board)` 의 set(1인 인덱스) == `get_valid_moves(actual_board)` 의 set. 현재 흑에선 미러 순열로 *다름* → 버그 증명. (A) 적용 시 canonical==actual이라 동일.
2. *흑 round-trip 테스트*: 흑 차례 여러 위치의 모든 합법수 `encode→decode` 왕복 일치(기존 테스트는 백만 검사).
3. *미러 불변성(gold standard)*: 위치 P(백 차례)와 그 미러 P′(흑 차례)는 동일 국면이므로, 같은 net에서 `value(P) == value(P′)`, 정책도 미러 대응. 수정 후 성립.

---

### 🔴 P0-2. 상태 저장이 히스토리를 버림

**문제.** 상태를 bare FEN 바이트로 저장하고 매 연산마다 `chess.Board(fen)`로 재생성한다. FEN엔 수순 히스토리가 없어 `is_repetition()`이 항상 False(→ 반복 plane 12·13 죽음), `b.outcome()`도 삼수동형·반복 무승부를 감지 못함. 무승부가 사실상 `max_moves` cap/스테일메이트로만 발생.

**방향성.**
- 상태에 **히스토리를 보존**: `(start_fen, 적용한 move 리스트)`를 저장하거나, 게임 루프에서 live `chess.Board`를 유지하며 push/pop. 재구성 시 `Board(start_fen)` + moves로 복원해 `is_repetition`/`outcome`이 동작하게.
- 또는 **게임 루프 레벨에서 위치 카운터**(transposition key의 `Counter`)를 유지해 삼수동형을 직접 판정하고 env에 전달.
- *반복 plane*은 위 히스토리로 정확히 채우거나, 당장 어려우면 **plane을 제거**해 "죽은 0 plane"이 net을 오도하지 않게. (50수 규칙은 `halfmove_clock`이 FEN에 있어 `outcome(claim_draw=True)`로 처리 가능 — 단 삼수동형은 히스토리 필요.)

**검증.**
- *삼수동형 테스트*(현재 실패 → 통과): 나이트를 3회 왕복시키는 짧은 수순 후 `get_game_ended`가 무승부 반환.
- *반복 plane 테스트*: 위치 반복 시 plane 12/13이 1이 됨(또는 plane을 제거했다면 NUM_INPUT_PLANES·net 입력 차원 일관성 테스트).

---

### 🟠 P1-1. 흑 underpromotion decode 오류

**문제.** `_decode_move_plane`의 underpromotion은 항상 `d_rank=+1`을 반환한다. 흑 폰(rank1→rank0 승급) decode가 `to_rank=from_rank+1`(뒤로 가는 불법수)를 만들어 `get_next_state`의 `assert move in b.legal_moves`에서 크래시할 수 있다. encode엔 흑 fallback이 있어 비대칭.

**방향성.** `decode_action`에서 underpromotion일 때 `board.turn`을 보고 `d_rank` 부호를 결정(백 +1, 흑 −1), `to_rank`가 해당 색의 마지막 랭크(백 7, 흑 0)가 되도록. P0-1을 (A)로 갈 경우 흑이 실제 좌표로 등장하므로 이 수정이 필수.

**검증.** *양색 승급 round-trip 테스트*: 백 폰 7랭크 + 흑 폰 2랭크가 있는 FEN에서 underpromotion 포함 전 합법수 `encode→decode` 왕복 일치(현재 흑에서 실패).

---

### 🟠 P1-2. 테스트 거짓 안심 → 게이트 추가

**문제.** `test_mcts.py`는 전부 player=+1, `test_encoding.py` round-trip은 백만 검사, end-to-end 학습 테스트(M2 게이트)·perft 부재. 그래서 P0-1·P1-1이 구조적으로 안 잡힌다.

**방향성 — 다음 테스트를 추가.**
1. **end-to-end "TicTacToe 완벽 학습"**: 실제 바깥 루프(self-play→buffer→trainer)를 몇 iteration 돌린 뒤, 결과 net+MCTS가 랜덤 상대에 무패·최적 무승부·알려진 위치 정답. → *파이프라인 전체*가 도는지 증명(단위 테스트가 못 하는 부분).
2. **흑 차례 체스 테스트**: P0-1의 프레임 일관성·미러 불변성·흑 round-trip.
3. **perft / mask 일치**: 무작위 위치(흑·승급 포함) 다수에서 `get_legal_move_mask` 가 `board.legal_moves`와 정확히 일치, 표준 위치 perft(depth 1) 노드 수 대조.

**검증.** 위 테스트가 (수정 전) 해당 버그에서 빨간불 → (수정 후) 모두 초록불. 1번은 회귀 가드로 상시 유지.

---

### 🟡 P2-1. Throughput

**문제.** `net.predict`가 시뮬레이션마다 단일 위치 평가(배치 없음), 매 노드 연산이 FEN→Board 재파싱 + `is_repetition` 스캔 → 체스에서 매우 느림. 구현 계획서의 binding constraint(batched self-play) 미반영.

**방향성 (정확성 수정 *후*에).** ① 여러 self-play 게임을 병렬로 돌려 leaf 평가를 **배치**로 GPU에 전달. ② 노드마다 FEN 재파싱 대신 Board 객체/증분 갱신 캐싱. ③ `is_repetition`을 매 leaf 호출하지 말고 유지 카운터 사용(P0-2와 연계). 과제 규모면 최소한 ①·② 만으로도 큰 개선.

**검증.** 수정 전후 games/sec 측정·목표 도달; 동시에 정확성 테스트(P0/P1) 전부 통과 유지(회귀 없음).

---

### ⚪ P2-2. 기타 + 미확인 항목

**기타(낮음).**
- net 기본 `10×128`이 계획(8×96)과 다름 → config 정렬 또는 정당화. (검증: config 값 확인.)
- value head `dropout=0.3` — AZ는 dropout 미사용 → 제거 검토. (검증: 제거 후 학습 안정성.)
- `_select_child`의 `sqrt(node.visit_count)`가 root 첫 시뮬레이션에서 0 → 첫 sim이 prior 무시(자기 수정되나 비효율). (검증: 첫 sim에서 최고 prior가 존중되는지 단위 테스트, 선택.)
- `trainer.py`의 미사용 `nn.CrossEntropyLoss()` dead code 제거(실제 손실은 수동 soft-CE로 올바름).

**미확인 — 감사 필요(버그 단정 아님).**
- `scripts/train.py`의 **바깥 반복 루프**(self-play→buffer→train→반복·체크포인트)가 올바로 엮였는지 미확인. → P1-2의 end-to-end 테스트가 이를 사실상 검증.
- `analysis/piece_values.py`의 **회귀 식별성**: 모든 패턴이 9점 고정이라 디자인 행렬 rank가 제한됨 → *절대* 기물값이 식별되는지 확인. (검증: 알려진 기물값을 심은 합성 로그에서 추정기가 식별 가능한 부분공간을 복원하는지 테스트.)
- `arena.py`/`track_b/lc0_runner.py`/`gamelog.py` 미감사.

---

## 3. 권장 수정 순서 (Phased Ladder)

| 단계 | 작업 | 통과 게이트 |
|------|------|-------------|
| **Phase 0 — 안전망** | P0-1·P0-2·P1-1 재현 실패 테스트 + end-to-end TTT 학습 + perft/mask 테스트 작성 | 의도한 버그에서 **빨간불 확인** |
| **Phase 1 — P0 수정** | P0-1(canonical) → P0-2(히스토리) 수정 | 프레임 일관성·미러 불변성·삼수동형 테스트 초록불 |
| **Phase 2 — P1 + 파이프라인** | P1-1(승급) 수정; **end-to-end TTT 완벽 학습 실제 통과** | TTT 학습 게이트 통과(바깥 루프 검증 포함) |
| **Phase 3 — 체스 검증** | 단일 핸디캡 조건으로 소규모 체스 학습; lc0/알려진 평가와 sanity; **흑이 정상적으로 두는지** 확인 | 흑·백 모두 합리적 플레이; loss↓ |
| **Phase 4 — P2** | throughput(배치·캐싱), 기타, 분석 식별성 | games/sec 목표; 회귀 없음 |

**핵심:** Phase 1 전에 절대 체스 학습 결과를 신뢰하지 말 것. Phase 2의 "TTT 완벽 학습"이 통과하기 전엔 파이프라인이 검증된 게 아니다.

---

## 4. 회귀 방지 / 마스터 검증

- **회귀 가드**: "TTT 완벽 학습", 프레임 일관성, 양색 round-trip, perft/mask, 삼수동형 테스트를 CI(또는 `pytest tests/`)에 상시 포함. 코어를 건드릴 때마다 실행.
- **마스터 검증(실험 착수 조건)**: ① 위 테스트 전부 초록 ② 단일 조건 체스에서 흑·백 모두 합리적 플레이 + lc0/통념과 sanity 일치 ③ **체크포인트·로그가 남는 실제 end-to-end 성공 run**(현재 부재) ④ (Track A 본학습 시) 모든 우선 패턴에서 약 baseline 대비 검증 게이트 통과.
- **결정성**: 고정 시드로 self-play 1게임·arena 1게임 재현 가능해야 디버깅이 성립.

---

*다음 단계 후보: (a) Phase 0의 재현 실패 테스트들을 실제 pytest 코드로 작성, (b) P0-1 (A) 방식 수정 패치 초안, (c) P0-2 히스토리 보존 리팩터 설계. 어디든 바로 들어갈 수 있음.*
