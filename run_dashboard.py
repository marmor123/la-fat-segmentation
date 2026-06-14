#!/usr/bin/env python3
"""Thin launcher script for the LA Fat interactive dashboard.

Usage
-----
    python run_dashboard.py                    # uses default ./outputs
    python run_dashboard.py --output-dir /path/to/outputs

The dashboard is served at http://localhost:5006 by default.  No browser
window is auto-opened.
"""

from __future__ import annotations

import argparse
import os
import sys


def _resolve_output_dir(candidate: str) -> str:
    """Return the absolute path to *candidate*, resolving relative paths."""
    resolved = os.path.abspath(candidate)
    if not os.path.isdir(resolved):
        print(f"Warning: output directory not found: {resolved}", file=sys.stderr)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the LA Fat interactive dashboard."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Path to the pipeline output directory (default: outputs)",
    )
    args = parser.parse_args()

    output_dir = _resolve_output_dir(args.output_dir)

    from la_fat.interactive_dashboard import create_dashboard

    dashboard = create_dashboard(output_dir)
    url = f"http://localhost:5006"
    print(f"Launching LA Fat Dashboard for: {output_dir}")
    print(f"Dashboard URL: {url}")
    print("Close the terminal or press Ctrl+C to stop.")

    # pn.serve starts the Bokeh server — does NOT auto-open a browser.
    import panel as pn

    pn.serve(dashboard, address="0.0.0.0", port=5006, show=False)


if __name__ == "__main__":
    main()
