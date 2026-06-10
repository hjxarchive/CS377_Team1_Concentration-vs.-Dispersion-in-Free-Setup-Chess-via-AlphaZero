import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handichess.track_b.lc0_runner import Lc0Runner

print("Running Track B Smoke Test...")

log_path = "scratch/track_b_log.jsonl"
if os.path.exists(log_path):
    os.remove(log_path)

# Try with default 'lc0' in PATH
runner = Lc0Runner(engine_path="lc0", nodes=10)
try:
    # Test just 1 pattern (2 games total) for speed
    runner.run_pattern(pattern_id="rook_4pawns", num_games=2, log=__import__('handichess.common.gamelog').common.gamelog.GameLog(log_path))
    
    print(f"\nChecking JSONL Log at {log_path}:")
    with open(log_path, 'r') as f:
        for line in f:
            print(line.strip())
except Exception as e:
    print(f"\nTrack B Smoke Test Failed. Reason: {e}")
    print("This is expected if the 'lc0' engine binary is not installed locally.")
