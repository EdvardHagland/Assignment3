#!/usr/bin/env python3
"""
Repair an already-generated exploratory HTML report so it works as a standalone
local file.

This script:
1. moves the embedded Plotly library into <head> so Plotly exists before any
   Plotly.newPlot(...) calls run; and
2. removes the external Google Fonts import so the file is more self-contained.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PLOTLY_SCRIPT_END = "</script>"
GOOGLE_FONTS_IMPORT = '@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:wght@400;600;700&display=swap");'
PLOTLY_MARKERS = (
    "!function(t,e)",
    "plotly.js",
    "window.Plotly",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair a standalone exploratory report HTML file.")
    parser.add_argument("input_html", help="Path to the generated HTML file.")
    parser.add_argument(
        "--output-html",
        default="",
        help="Optional output path. Defaults to overwriting the input file in place.",
    )
    return parser.parse_args()


def extract_plotly_bundle(html_text: str) -> tuple[str, str]:
    body_close = html_text.rfind("</body>")
    search_space = html_text if body_close == -1 else html_text[:body_close]

    for marker in PLOTLY_MARKERS:
        marker_pos = search_space.rfind(marker)
        if marker_pos == -1:
            continue
        start = search_space.rfind("<script", 0, marker_pos)
        if start == -1:
            continue
        script_tag_close = search_space.find(">", start)
        if script_tag_close == -1 or script_tag_close > marker_pos:
            continue
        end = search_space.find(PLOTLY_SCRIPT_END, marker_pos)
        if end == -1:
            continue
        end += len(PLOTLY_SCRIPT_END)
        return html_text[start:end], html_text[:start] + html_text[end:]

    raise RuntimeError("Could not find embedded Plotly bundle in the HTML file.")


def repair_html(html_text: str) -> str:
    plotly_bundle, without_bundle = extract_plotly_bundle(html_text)
    without_bundle = without_bundle.replace(GOOGLE_FONTS_IMPORT, "")
    if plotly_bundle in without_bundle:
        return without_bundle
    head_close = without_bundle.find("</head>")
    if head_close == -1:
        raise RuntimeError("Could not find </head> in the HTML file.")
    return without_bundle[:head_close] + plotly_bundle + "\n" + without_bundle[head_close:]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_html)
    output_path = Path(args.output_html) if args.output_html else input_path

    html_text = input_path.read_text(encoding="utf-8")
    repaired = repair_html(html_text)
    output_path.write_text(repaired, encoding="utf-8")
    print(f"Repaired standalone HTML written to: {output_path}")


if __name__ == "__main__":
    main()
