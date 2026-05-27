#!/usr/bin/env python3
"""
Floor Pilot for Pawn-heavy bundles (bishop_6pawns, knight_6pawns).
These bundles leave the Q side with very few pawns, risking extreme imbalance.
This script evaluates 10 games per pattern using LC0 to see if win rates fall 
outside the 0.3~0.7 threshold.
"""

import sys
import logging
from collections import defaultdict
from handichess.track_b.lc0_runner import LC0Runner
from handichess.common.gamelog import GameLog

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    target_patterns = ["bishop_6pawns", "knight_6pawns"]
    num_games_per_pattern = 10
    nodes = 100
    
    logger.info(f"Starting Floor Pilot for: {target_patterns}")
    
    # Try to init LC0 Runner
    try:
        runner = LC0Runner(engine_path="lc0", nodes=nodes)
    except Exception as e:
        logger.error(f"Could not initialize LC0 (ensure 'lc0' is in PATH). Error: {e}")
        logger.info("Skipping floor pilot execution. Please run this on the main cluster.")
        sys.exit(0)
        
    game_log = GameLog("runs/floor_pilot.jsonl")
    
    results = defaultdict(list)
    
    for pid in target_patterns:
        logger.info(f"--- Testing {pid} ---")
        for i in range(num_games_per_pattern):
            # Alternate Q side
            q_side = "white" if i % 2 == 0 else "black"
            try:
                record = runner.evaluate_matchup(pid, q_side=q_side)
                game_log.write(record)
                results[pid].append(record.result_score)
                logger.info(f"[{pid}] Game {i+1}/{num_games_per_pattern} | Q={q_side} -> Score={record.result_score}")
            except Exception as e:
                logger.error(f"Error during evaluation of {pid}: {e}")
                
        # Analyze distribution
        scores = results[pid]
        if scores:
            win_rate = sum(scores) / len(scores)
            logger.info(f"[PILOT RESULT] {pid}: Win rate for Q-side = {win_rate:.2f} ({len(scores)} games)")
            if not (0.3 <= win_rate <= 0.7):
                logger.warning(f"🚨 {pid} is OUTSIDE the 0.3~0.7 bounds (WR={win_rate:.2f}). Consider removing or isolating from main analysis.")
            else:
                logger.info(f"✅ {pid} is within acceptable bounds.")
        else:
            logger.warning(f"No games completed for {pid}.")

if __name__ == "__main__":
    main()
