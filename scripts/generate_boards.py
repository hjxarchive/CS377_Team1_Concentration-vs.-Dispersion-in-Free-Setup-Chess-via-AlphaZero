#!/usr/bin/env python3
import chess
import chess.svg
from pathlib import Path
import sys
import re

sys.path.append(str(Path(__file__).parent.parent))
from handichess.common.handicap import get_patterns, make_matchup_board

OUT_DIR = Path("figures/boards")
OUT_DIR.mkdir(parents=True, exist_ok=True)

patterns = get_patterns()

custom_style = """
.square.light { fill: #ffffff; }
.square.dark { fill: #cccccc; }
.board-border { stroke: #000000; stroke-width: 3px; fill: none; }
text.coordinate { fill: #000000; font-weight: bold; font-family: sans-serif; font-size: 14px; }
"""

for p in patterns:
    board = make_matchup_board(p, noq_color=chess.BLACK)
    
    svg_data = chess.svg.board(
        board=board,
        orientation=chess.WHITE,
        size=400,
        coordinates=True,
        style=custom_style
    )
    
    # Inject a white background rect behind everything so the coordinate margin is explicitly white
    # Also inject the whiteOutline filter just in case, but we don't apply it to <use> anymore 
    # since solid gray background provides enough contrast.
    bg_rect = '<rect width="100%" height="100%" fill="#ffffff" />'
    svg_data = re.sub(r'(<svg[^>]*>)', r'\1\n' + bg_rect, svg_data, count=1)
        
    with open(OUT_DIR / f"{p.pattern_id}.svg", "w") as f:
        f.write(svg_data)

print(f"Successfully generated SVGs with white margin and gray dark squares in {OUT_DIR}/")
