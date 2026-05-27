# 패치 계획서 — 마이그레이션 후 잔여 항목 (본 run 착수 전)

**대상:** HandiChess (Design B 마이그레이션 완료 커밋 기준) · **Team 1**
**원칙:** Red → Fix → Green. 클러스터 본 run(예: `--iterations 100 --games-per-iter 1024`) *전에* 아래를 모두 green으로.

---

## 0. 검증 결과 요약 (직접 확인)

**이미 올바르게 반영됨(통과):**
- P0-1 수정 — `get_canonical_form` no-op(+근거 docstring). ✅
- P0-2 수정 — 상태 = `FEN + 전체 수순(UCI)`, replay로 히스토리 복원. ✅
- queen 패턴 제거 + `noq_removals`/`q_removals` 분리 + 양측 9점·30점 assert. ✅
- `test_matchup.py` 통과 — 격리 불변식(퀸 0/1, 양측 30, 차이 벡터, 합법성) 7패턴×양색. ✅
- 생성 FEN 정확(미러·캐슬링권 갱신 포함), floor 점검 스크립트 존재. ✅

**남은 항목 → 본 문서가 다룸:**
- 🔴 `test_handicap.py`(stale)가 전체 스위트 collection을 깨뜨림.
- 🟠 `test_chess_gate.py`가 P0-1 회귀를 *실제로는 못 잡음*(거짓 안심).
- 🟠 런타임 게이트(TTT 학습, 흑 측 sanity) 미검증 — torch 환경 필요.
- 🟡 폰-heavy 묶음(bishop_6pawns·knight_6pawns) 변별성 미확인.
- 🟡 본 run 위생(suite green, 로깅, 시드).

---

## 1. 패치 목록

| ID | 항목 | 심각도 | 비고 |
|----|------|--------|------|
| **PATCH-1** | stale `test_handicap.py` 정리 | 🔴 blocker | 안 고치면 `pytest tests/` collection 에러 |
| **PATCH-2** | `test_chess_gate.py` 강화(P0-1 회귀 가드) | 🟠 | 지금은 미러 버그가 부활해도 통과함 |
| **PATCH-3** | 런타임 게이트(TTT 학습 + 소규모 체스 sanity) | 🟠 | torch 환경에서 실행 |
| **PATCH-4** | 폰-heavy 묶음 변별성 파일럿 | 🟡 | 결과로 분석 포함/제외 결정 |
| **PATCH-5** | 본 run 위생(green·로깅·시드) | 🟡 | DoD 항목 |

---

## 2. 패치별 상세 (문제 / 변경 / 검증)

### 🔴 PATCH-1. stale `test_handicap.py` 정리

**문제.** `from handichess.common.handicap import make_handicap_board` (마이그레이션에서 제거된 API) → `pytest tests/` 가 collection 단계에서 ImportError로 중단. 즉 "Error fixed, ready"라지만 *전체 스위트는 아직 green이 아님*.

**변경.** 먼저 내용을 확인 후:
- (권장) **삭제** — 점수합·합법성·기물 검사가 `test_matchup.py`에 흡수됐다면 중복.
- 고유 커버리지(예: `count_material`, `material_balance`, FEN 캐스팅권 등)가 있다면 그 부분만 `test_matchup.py`로 **포팅**.

**검증.** `pytest tests/ --collect-only` 에러 없음 → `pytest tests/` 가 collection을 통과(개별 통과/실패와 무관하게 import 단계는 깨끗).

---

### 🟠 PATCH-2. `test_chess_gate.py` 강화 — P0-1 회귀 가드

**문제.** 현재 테스트는 (a) 소유권 plane 카운트, (b) mask 정렬, (c) 무크래시만 본다. **미러 버그가 부활해도 전부 통과**한다(미러는 기물 *개수*를 보존하고, mask가 적용되므로 prob은 여전히 합법수에만 실림). 즉 이름(mirror invariant)과 달리 *P0-1을 못 잡는 거짓 안심 테스트*. (no-op canonical에선 "미러 불변성"은 애초에 맞는 속성도 아님 — net이 색 plane으로 방향을 본다.)

**변경 — 미러 버그가 부활하면 *실패*하는 테스트를 추가.**
- **(싸고 확실) 프레임 일관성 테스트(torch 불필요):** 흑 차례 위치에서
  `get_legal_move_mask(_state_to_board(get_canonical_form(state, -1)))` == `get_valid_moves(state, -1)`.
  no-op면 canonical==actual이라 동일; 누가 `get_canonical_form`에 `b.mirror()`를 되살리면 미러 순열로 *달라져 실패*.
- **(강하지만 torch 필요) prior-프레임 테스트:** 흑 root를 `_expand` 후, 각 합법 action `a`에 대해 `child.prior` ≈ `softmax(net policy on encode_board(actual board))[a]`. 미러 시 prior가 다른 좌표에서 와서 불일치 → 실패.
- 테스트 이름도 `test_canonical_is_noop_actual_frame` 등으로 정정.

**검증 (가드가 진짜 가드인지 증명).** `get_canonical_form`을 *임시로* `return self._state_to_board(...).mirror()...` 로 되돌려 새 테스트가 **RED**가 되는지 확인 → 다시 no-op로 복원하면 **GREEN**. (테스트가 의도한 버그를 실제로 잡는지 검증하는 단계.)

---

### 🟠 PATCH-3. 런타임 게이트 — TTT 학습 + 소규모 체스 sanity

**문제.** 단위 테스트로는 "파이프라인이 학습하는가 / 흑이 합리적으로 두는가"를 증명 못 한다. (torch 환경 필요, 본 환경에선 미실행.)

**변경/실행.**
- `test_pipeline.py`(TTT end-to-end): 단순 무크래시가 아니라 **학습 후 랜덤 상대 무패 / 알려진 위치 정답**을 assert하도록 강화(이미 그렇다면 확인만).
- **소규모 체스 sanity 스크립트/레시피:** `train.py --game chess --iterations 3 --games-per-iter 16 --simulations 100` 로 돌려 loss 감소 + **흑·백 둘 다 합법·합리적으로 두는지**(약 baseline/lc0 대비) 확인. no-op canonical이라 net이 양방향을 배워야 하니, *흑이 약하면 버그가 아니라 학습 부족*일 수 있음 → 최소한 *합법·비자살적*인지부터.

**검증.** `pytest tests/test_pipeline.py tests/test_chess_gate.py tests/test_mcts.py -v` 통과 + 소규모 체스 run에서 loss↓ & 흑 정상.

---

### 🟡 PATCH-4. 폰-heavy 묶음 변별성 파일럿

**문제.** `bishop_6pawns`/`knight_6pawns`는 Q측 폰이 2개만 남아(검증 FEN에서 확인: `…/6PP/…`) 한쪽으로 쏠려 floor가 재발할 수 있음 → 변별 신호 없음.

**변경/실행.** lc0(동일 nodes)로 각 매치업 **파일럿 ~100게임**(양색) → 점수가 변별 구간(대략 0.3~0.7) 밖(예: >90/10)이면:
- 메인 가산성 분석에서 **제외하거나 "극단 분산" 별도 카테고리**로 표기,
- 또는 그 사실 자체를 결과로 보고("6폰 분산은 퀸 대비 명백히 열세").

**검증.** 매치업별 파일럿 승/무율 표 → 본 run에 포함할 묶음 목록 확정.

---

### 🟡 PATCH-5. 본 run 위생

**변경.**
- PATCH-1 후 `pytest tests/` **전부 green** 확인.
- `gamelog`: `noq_side` + **`bundle_id`** + **Q측 색** + **점수 기준 고정**(예: 항상 *Q side(집중) 점수*)이 기록되는지 — 흑백 비대칭·랭킹 분석에 필수.
- **시드 고정**: self-play 1게임·arena 1게임이 재현되는지(디버깅 전제).
- lc0(Track B)는 **동일 nodes**(B는 floor effect 없음 → 강도 비대칭 불필요).

**검증.** 로그 1건을 열어 필드 완비 확인; 고정 시드 2회 실행 결과 일치.

---

## 3. 적용 순서 + 게이트

| 순서 | 패치 | 통과 게이트 |
|------|------|-------------|
| 1 | PATCH-1 | `pytest tests/` collection 에러 0 |
| 2 | PATCH-2 | 새 가드가 RED→(no-op 복원)GREEN; 회귀 가드 확보 |
| 3 | PATCH-3 | TTT 학습 통과 + 소규모 체스에서 흑·백 정상 |
| 4 | PATCH-4 | 변별 묶음 목록 확정(폰-heavy 처리 결정) |
| 5 | PATCH-5 | 전체 green + 로깅/시드 위생 |

---

## 4. 본 run 착수 조건 (Definition of Done)

다음이 *모두* 충족되면 4-GPU 본 run을 신뢰할 수 있다:
1. `pytest tests/` 전부 green(stale 제거 + 강화된 gate 포함).
2. P0-1 회귀 가드가 *실제로 버그를 잡음*을 증명(PATCH-2 RED/GREEN 확인).
3. 소규모 체스 run에서 loss↓ & 흑·백 합리적 플레이.
4. 파일럿으로 변별 가능한 매치업 목록 확정(폰-heavy 처리 결정).
5. 로깅(bundle_id·색·점수기준)·시드 위생 완료, lc0 동일 nodes.

> 그 다음에야: Track B(lc0, 동일 강도, 매치업당 수백~천판, 오프닝 샘플링, 2 depth 레벨, 사전등록)로 메인 결과 → Track A는 corroboration.

---

*다음 단계 후보: (a) PATCH-1 삭제/포팅 디프, (b) PATCH-2 강화 테스트 코드(프레임 일관성 + prior-프레임), (c) PATCH-3 소규모 체스 sanity 스크립트. 어디든 바로 작성 가능.*
