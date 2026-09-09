"""Cut the coloured cubes out of img/ into RGBA crops for scene synthesis.

    uv run scripts/extract_blocks.py
    uv run scripts/extract_blocks.py --source img --out data/blocks
"""

import argparse

from so101.dataset import extract_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="img")
    parser.add_argument("--out", default="data/blocks")
    args = parser.parse_args()

    counts = extract_blocks(args.source, args.out)
    print(f"\n{sum(counts.values())} crops written to {args.out}")


if __name__ == "__main__":
    main()
