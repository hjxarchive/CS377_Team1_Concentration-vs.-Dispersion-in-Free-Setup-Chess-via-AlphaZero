import chess
from handichess.common.handicap import get_pattern_by_id, make_matchup_board, count_material

def render_board(board, title):
    print("=" * 40)
    print(title)
    print("=" * 40)
    print(board)
    print("\nFEN:", board.fen())
    print("White Material:", count_material(board, chess.WHITE))
    print("Black Material:", count_material(board, chess.BLACK))
    print("-" * 40, "\n")

def main():
    pattern = get_pattern_by_id("bishop_6pawns")
    
    # 시나리오 1: 백이 NoQ (백은 퀸 제거, 흑은 B+6P 제거)
    board_w_noq = make_matchup_board(pattern, chess.WHITE)
    render_board(board_w_noq, "Scenario 1: White is NoQ (White removes Queen, Black removes B+6P)")
    
    # 시나리오 2: 흑이 NoQ (흑은 퀸 제거, 백은 B+6P 제거)
    board_b_noq = make_matchup_board(pattern, chess.BLACK)
    render_board(board_b_noq, "Scenario 2: Black is NoQ (Black removes Queen, White removes B+6P)")

if __name__ == "__main__":
    main()
