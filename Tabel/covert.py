#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a Markdown table in a .md file to an Excel .xlsx file.

Usage:
  python md_to_excel.py /path/to/summary_table.md -o summary_table.xlsx
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def extract_first_markdown_table(md_text: str) -> str:
    """
    Extract the first markdown table block (|...| with a separator row).
    Returns the table text, or raises ValueError if not found.
    """
    lines = md_text.splitlines()

    table_lines = []
    in_table = False

    # A markdown table usually has:
    # header: | a | b |
    # sep:    |---|---|
    sep_re = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")

    for i in range(len(lines) - 1):
        line = lines[i]
        next_line = lines[i + 1]

        if (not in_table) and line.strip().startswith("|") and sep_re.match(next_line.strip()):
            # start table
            in_table = True
            table_lines.append(line.rstrip())
            table_lines.append(next_line.rstrip())
            continue

        if in_table:
            # continue while lines look like table rows
            if line.strip().startswith("|"):
                # avoid duplicating the header already appended
                if lines[i - 1].rstrip() != line.rstrip():
                    table_lines.append(line.rstrip())
            else:
                break

    if not table_lines:
        raise ValueError("No Markdown table found in the input file.")

    # Deduplicate potential double-add and remove empty lines
    table_lines = [l for l in table_lines if l.strip()]
    return "\n".join(table_lines)


def markdown_table_to_df(table_text: str) -> pd.DataFrame:
    """
    Convert markdown table text to a pandas DataFrame.
    """
    rows = []
    for line in table_text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # trim leading/trailing |
        parts = [c.strip() for c in line.strip("|").split("|")]
        rows.append(parts)

    if len(rows) < 2:
        raise ValueError("Table is too short to parse (need header + rows).")

    header = rows[0]
    sep = rows[1]

    # if second row is separator (---), drop it
    def is_sep_row(r):
        return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in r)

    data_rows = rows[2:] if is_sep_row(sep) else rows[1:]

    # normalize row lengths
    max_cols = len(header)
    norm = []
    for r in data_rows:
        if len(r) < max_cols:
            r = r + [""] * (max_cols - len(r))
        elif len(r) > max_cols:
            r = r[:max_cols]
        norm.append(r)

    df = pd.DataFrame(norm, columns=header)

    # try to cast numeric columns where possible
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).strip())
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md_path", type=str, help="Path to markdown file")
    ap.add_argument("-o", "--output", type=str, default=None, help="Output xlsx path")
    ap.add_argument("--sheet", type=str, default="Sheet1", help="Excel sheet name")
    args = ap.parse_args()

    md_path = Path(args.md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Input file not found: {md_path}")

    out_path = Path(args.output) if args.output else md_path.with_suffix(".xlsx")

    md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    table_text = extract_first_markdown_table(md_text)
    df = markdown_table_to_df(table_text)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=args.sheet)

        # basic formatting: autofit column widths
        ws = writer.sheets[args.sheet]
        for col_cells in ws.columns:
            max_len = 0
            col_letter = col_cells[0].column_letter
            for cell in col_cells:
                val = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(val))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    print(f"Saved: {out_path}  (rows={len(df)}, cols={len(df.columns)})")


if __name__ == "__main__":
    main()