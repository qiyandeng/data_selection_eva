#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate PPL-computed JSONL files and select data by PPL descending.

Usage:
    python data-select.py \
        --input_prefix ppl \
        --select_ratio 0.01 \
        --output_path ppl-0.01.jsonl
"""

import argparse, json, glob
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Select data by PPL descending")
    parser.add_argument("--input_prefix", type=str, required=True,
                        help="Prefix of PPL-computed output files, e.g., 'ppl_out' matches 'ppl_out_rank0.jsonl', etc.")
    parser.add_argument("--select_ratio", type=float, required=True,
                        help="Ratio of data to select (0 < ratio <= 1)")
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Path to save selected output JSONL")
    return parser.parse_args()

def load_all_entries(input_prefix):
    files = glob.glob(f"{input_prefix}_rank*.jsonl")
    all_data = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # 支持列表或标量 PPL（代码中保存为 list）
                    ppl_val = item["PPL"][0] if isinstance(item["PPL"], list) else item["PPL"]
                    item["_PPL"] = ppl_val
                    all_data.append(item)
                except:
                    continue
    return all_data

def main():
    args = parse_args()
    assert 0 < args.select_ratio <= 1, "select_ratio must be in (0, 1]"

    print(f"Loading files with prefix: {args.input_prefix}_rank*.jsonl")
    data = load_all_entries(args.input_prefix)

    print(f"Loaded {len(data)} entries. Sorting by PPL descending...")
    data.sort(key=lambda x: x["_PPL"], reverse=True)

    n_select = int(len(data) * args.select_ratio)
    selected = data[:n_select]

    print(f"Selecting top {n_select} entries (ratio = {args.select_ratio})")
    with args.output_path.open("w", encoding="utf-8") as fout:
        for entry in selected:
            del entry["_PPL"]
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved selected entries to: {args.output_path}")

if __name__ == "__main__":
    main()
