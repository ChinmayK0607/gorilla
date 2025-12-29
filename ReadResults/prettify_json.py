#!/usr/bin/env python3
"""
Pretty-print JSON result files for easier inspection.

Usage:
    python prettify_json.py [path/to/file.json] [--output OUTPUT] [--in-place] [--sort-keys]

Defaults to the BFCL multi-turn miss-func result file if no path is provided.
"""
import argparse
import json
from pathlib import Path


DEFAULT_RESULT_PATH = Path(__file__).resolve().parent / "Qwen_Qwen2.5-7B-Instruct-FC" / "multi_turn" / "BFCL_v4_multi_turn_miss_func_result.json"


def prettify_json(
    input_path: Path, output_path: Path | None = None, in_place: bool = False, sort_keys: bool = False
) -> None:
    data = json.loads(input_path.read_text())
    pretty = json.dumps(data, indent=2, ensure_ascii=True, sort_keys=sort_keys)

    if in_place:
        input_path.write_text(pretty + "\n")
    elif output_path is not None:
        output_path.write_text(pretty + "\n")
    else:
        print(pretty)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prettify a JSON result file.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_RESULT_PATH),
        help="Path to the JSON file to prettify (defaults to the multi_turn miss_func result).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Optional path to write prettified JSON. Prints to stdout when omitted unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the input file with prettified JSON.",
    )
    parser.add_argument(
        "--sort-keys",
        action="store_true",
        help="Sort object keys alphabetically for stable output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    prettify_json(
        input_path=input_path,
        output_path=output_path,
        in_place=args.in_place,
        sort_keys=args.sort_keys,
    )


if __name__ == "__main__":
    main()
