"""Export nucleus/myotube measurements directly from this checkout."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fouri_quantification.outputs.measurements import make_measurements_h5ad, make_split_measurements_h5ad


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a compact nucleus/myotube analysis H5AD")
    parser.add_argument("--input", required=True, type=Path, help="Full myonis_4i_data.h5ad")
    parser.add_argument("--output", type=Path, help="Default: <input_stem>_measurements.h5ad beside input")
    parser.add_argument("--repeat", type=int, metavar="N",
                        help="Keep only replicate N in nucleus, myotube, sample and field QC tables")
    parser.add_argument("--split", action="store_true", help="Write separate nuclei and myotube H5ADs")
    parser.add_argument("--compact-myotubes", action="store_true",
                        help="With --split: keep direct myotube features and resident-nucleus mean/n_valid, dropping resident median/std and repeated source paths")
    args = parser.parse_args()
    if args.repeat is not None and args.repeat < 1:
        parser.error("--repeat must be a positive integer")
    if args.compact_myotubes and not args.split:
        parser.error("--compact-myotubes requires --split")
    if args.split and args.output is not None:
        parser.error("--output applies to the combined file; split files are named beside --input")
    if args.split:
        make_split_measurements_h5ad(args.input, compact_myotubes=args.compact_myotubes,
                                     repeat=args.repeat)
    else:
        make_measurements_h5ad(args.input, args.output, repeat=args.repeat)


if __name__ == "__main__":
    main()
