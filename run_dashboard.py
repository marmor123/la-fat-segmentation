#!/usr/bin/env python3
"""Thin launcher script for the zero-footprint LA Fat QA Dashboard.

Usage
-----
    python run_dashboard.py                    # opens default data/outputs/cohort_qa_dashboard.html
    python run_dashboard.py --output-dir /path/to/outputs
    python run_dashboard.py --patient 0674

Opens the standalone HTML5 QA Studio in the default web browser.
Zero runtime server, zero npm, 100% offline.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the LA Fat QA Studio in default browser."
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs",
        help="Path to pipeline output directory (default: data/outputs)",
    )
    parser.add_argument(
        "--patient",
        default=None,
        help="Open a specific patient's QA report (e.g. '0674')",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    if args.patient:
        target_html = os.path.join(output_dir, args.patient, "qa_report.html")
    else:
        target_html = os.path.join(output_dir, "cohort_qa_dashboard.html")
        if not os.path.isfile(target_html):
            # Check if there are patient subdirectories with qa_report.html
            for item in os.listdir(output_dir) if os.path.isdir(output_dir) else []:
                cand = os.path.join(output_dir, item, "qa_report.html")
                if os.path.isfile(cand):
                    target_html = cand
                    break

    if not os.path.isfile(target_html):
        print(f"[-] QA Dashboard HTML not found at: {target_html}", file=sys.stderr)
        print(f"[-] Run the pipeline first: python run_pipeline.py --patient <ID>", file=sys.stderr)
        sys.exit(1)

    print(f"[+] Opening QA Studio: {target_html}")
    webbrowser.open(f"file://{os.path.abspath(target_html)}")


if __name__ == "__main__":
    main()
