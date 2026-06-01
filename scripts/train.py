#!/usr/bin/env python3
"""
Train AlphaZero (Track A).

Supports both TicTacToe (for core verification) and chess.
"""

import argparse
import logging
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h{int(m)}m{int(s)}s"


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero")
    parser.add_argument("--game", type=str, default="tictactoe",
                        choices=["tictactoe", "chess"],
                        help="Game to train on")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--games-per-iter", type=int, default=256,
                        help="Self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=None,
                        help="MCTS simulations per move")
    parser.add_argument("--checkpoint-dir", type=str, default="runs/checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Handicap pattern (chess only)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (auto-detect if not given)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--max-moves", type=int, default=180,
                        help="Maximum half-moves per game (chess only)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    # Create game
    if args.game == "tictactoe":
        from handichess.track_a.game.tictactoe import TicTacToeGame
        game = TicTacToeGame()
        default_sims = 50
        mcts_config = {"dirichlet_alpha": 1.0, "temperature_threshold": 6}
        net_config = {"num_res_blocks": 4, "num_channels": 64}
    else:
        from handichess.track_a.game.chess_std import ChessGame
        from handichess.track_a.selfplay import create_matchup_start_states
        game = ChessGame(max_moves=args.max_moves)
        if args.pattern:
            # 특정한 패턴만 집중 학습
            start_states = create_matchup_start_states(ChessGame, [args.pattern], max_moves=args.max_moves)
        else:
            # 전체 7개 패턴 모두 섞어서 학습
            from handichess.common.handicap import get_patterns
            all_patterns = [p.pattern_id for p in get_patterns()]
            start_states = create_matchup_start_states(ChessGame, all_patterns, max_moves=args.max_moves)
        default_sims = 800
        mcts_config = {"dirichlet_alpha": 0.3, "temperature_threshold": 30}
        net_config = {"num_res_blocks": 10, "num_channels": 128}

    num_sims = args.simulations or default_sims
    mcts_config["num_simulations"] = num_sims

    # Create net
    from handichess.track_a.net import create_net_for_game
    net = create_net_for_game(game, net_config)
    logger.info(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Log training config summary
    logger.info(f"Config: game={args.game} iters={args.iterations} games/iter={args.games_per_iter} sims={num_sims} max_moves={args.max_moves}")

    # Create trainer
    from handichess.track_a.trainer import Trainer
    # Training config — chess needs larger batches for GPU saturation
    if args.game == "chess":
        train_config = {"epochs_per_iteration": 10, "batch_size": 512}
    else:
        train_config = {"epochs_per_iteration": 10, "batch_size": 64}

    trainer = Trainer(
        net, device=device,
        checkpoint_dir=args.checkpoint_dir,
        config=train_config,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Create self-play
    from handichess.track_a.selfplay import SelfPlay, ReplayBuffer
    selfplay = SelfPlay(
        game, net,
        mcts_config=mcts_config,
        selfplay_config={"max_moves": args.max_moves},
        device=device,
    )
    buffer = ReplayBuffer(max_size=200_000)

    # Timing accumulators
    total_train_start = time.time()
    iter_times = []  # wall-clock per iteration
    selfplay_times = []
    train_times = []

    remaining_iters = args.iterations - trainer.iteration
    logger.info(f"Starting training: {remaining_iters} iterations remaining")

    # Training loop
    for iteration in range(trainer.iteration, args.iterations):
        iter_start = time.time()
        iter_num = iteration + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iter_num}/{args.iterations}")
        logger.info(f"{'='*60}")

        # ── Phase 1: Self-play ──
        logger.info(f"[Phase 1/3] Self-play: {args.games_per_iter} games...")
        t0 = time.time()
        
        # start_states를 넘겨주면, 그 중 랜덤하게 뽑아 게임을 시작함
        examples = selfplay.generate_games(args.games_per_iter, start_states=start_states if args.game == "chess" else None)
        selfplay_elapsed = time.time() - t0
        selfplay_times.append(selfplay_elapsed)

        buffer.add(examples)
        avg_examples_per_game = len(examples) / max(1, args.games_per_iter)
        logger.info(
            f"  Self-play done: {len(examples)} examples in {_fmt_duration(selfplay_elapsed)} "
            f"({avg_examples_per_game:.0f} examples/game, buffer: {len(buffer)})"
        )

        # ── Phase 2: Training ──
        train_sample_size = min(len(buffer), 8192)
        logger.info(f"[Phase 2/3] Training: {train_sample_size} examples, {train_config.get('epochs_per_iteration', 10)} epochs, batch_size={train_config.get('batch_size', 256)}...")
        t0 = time.time()

        train_data = buffer.sample(train_sample_size)
        stats = trainer.train(train_data, iteration=iteration)
        train_elapsed = time.time() - t0
        train_times.append(train_elapsed)

        logger.info(
            f"  Training done: {_fmt_duration(train_elapsed)} | "
            f"loss={stats['total_loss']:.4f} (pi={stats['policy_loss']:.4f}, v={stats['value_loss']:.4f})"
        )

        # ── Phase 3: Checkpoint ──
        ckpt_elapsed = 0.0
        if (iter_num) % 10 == 0:
            logger.info(f"[Phase 3/3] Saving checkpoint...")
            t0 = time.time()
            trainer.save_checkpoint()
            ckpt_elapsed = time.time() - t0
            logger.info(f"  Checkpoint saved in {_fmt_duration(ckpt_elapsed)}")

        # ── Iteration Summary ──
        iter_elapsed = time.time() - iter_start
        iter_times.append(iter_elapsed)
        total_elapsed = time.time() - total_train_start
        completed = len(iter_times)
        remaining = args.iterations - iter_num
        avg_iter_time = sum(iter_times) / len(iter_times)
        eta = avg_iter_time * remaining

        selfplay_pct = selfplay_elapsed / max(1e-9, iter_elapsed) * 100
        train_pct = train_elapsed / max(1e-9, iter_elapsed) * 100

        logger.info(
            f"  ── Iter {iter_num} Summary ──  "
            f"total={_fmt_duration(iter_elapsed)} "
            f"(selfplay={_fmt_duration(selfplay_elapsed)} [{selfplay_pct:.0f}%] | "
            f"train={_fmt_duration(train_elapsed)} [{train_pct:.0f}%]"
            f"{f' | ckpt={_fmt_duration(ckpt_elapsed)}' if ckpt_elapsed > 0 else ''})"
        )
        logger.info(
            f"  ── Progress ──  "
            f"{completed}/{args.iterations} done | "
            f"elapsed={_fmt_duration(total_elapsed)} | "
            f"avg={_fmt_duration(avg_iter_time)}/iter | "
            f"ETA={_fmt_duration(eta)} ({remaining} iters left)"
        )

    # Final save
    logger.info(f"\n{'='*60}")
    trainer.save_checkpoint("final.pt")
    total_elapsed = time.time() - total_train_start

    # ── Grand Summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Total time:     {_fmt_duration(total_elapsed)}")
    logger.info(f"  Iterations:     {len(iter_times)}")
    if iter_times:
        logger.info(f"  Avg iter:       {_fmt_duration(sum(iter_times)/len(iter_times))}")
        logger.info(f"  Avg selfplay:   {_fmt_duration(sum(selfplay_times)/len(selfplay_times))}")
        logger.info(f"  Avg training:   {_fmt_duration(sum(train_times)/len(train_times))}")
        logger.info(f"  Selfplay total: {_fmt_duration(sum(selfplay_times))} ({sum(selfplay_times)/total_elapsed*100:.0f}%)")
        logger.info(f"  Training total: {_fmt_duration(sum(train_times))} ({sum(train_times)/total_elapsed*100:.0f}%)")


if __name__ == "__main__":
    main()
