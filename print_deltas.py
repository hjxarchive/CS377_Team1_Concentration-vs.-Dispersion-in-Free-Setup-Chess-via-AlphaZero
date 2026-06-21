import sys
from pathlib import Path
from scripts.generate_figures import load_jsonl, wdl, score
RESULTS_DIR = Path("runs/results")
LABELS = {
    "rook_bishop_pawn":     "R+B+P",
    "rook_knight_pawn":     "R+N+P",
    "bishop_bishop_knight": "B+B+N",
    "rook_4pawns":          "R+4P",
    "bishop_knight_3pawns": "B+N+3P",
    "bishop_6pawns":        "B+6P",
    "knight_6pawns":        "N+6P",
}
for p in LABELS.keys():
    fp = RESULTS_DIR / f"track_b_stochastic_{p}.jsonl"
    if fp.exists():
        g = load_jsonl(fp)
        wg = [x for x in g if x["noq_side"] == "black"]
        bg = [x for x in g if x["noq_side"] == "white"]
        qw, qd, ql = wdl(wg)
        bw, bd, bl = wdl(bg)
        q_ws = score(qw, qd, ql)
        q_bs = score(bl, bd, bw)
        print(f"{LABELS[p]}: White={q_ws:.3f}, Black={q_bs:.3f}, Delta={q_ws - q_bs:+.3f}")
