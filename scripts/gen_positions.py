#!/usr/bin/env python3
"""Generate all handicap starting positions and print/save them."""

import argparse
import json
import sys

from handichess.common.handicap import generate_all_positions


def main():
    parser = argparse.ArgumentParser(description="Generate handicap FEN positions")
    parser.add_argument("--phase", type=int, default=None, help="Phase filter (1 or 2)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    positions = generate_all_positions(phase=args.phase)

    print(f"Generated {len(positions)} positions:\n")
    for pos in positions:
        side = "WHITE" if pos.handicap_side else "BLACK"
        print(f"[{pos.pattern_id}] handicap={pos.to_dict()['handicap_side'].upper()}")
        print(f"  FEN: {pos.fen}")
        print(f"  Removed: {pos.material_diff}")
        print()

    if args.output:
        data = [pos.to_dict() for pos in positions]
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
