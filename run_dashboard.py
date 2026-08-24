#!/usr/bin/env python3
"""Thin launcher script for the zero-footprint LA Fat QA Dashboard.

Usage::

    python run_dashboard.py                    # opens default cohort QA dashboard
    python run_dashboard.py --patient 0674     # opens patient 0674 QA report
    python run_dashboard.py --output-dir /dir  # opens QA dashboard in specified folder

Opens the standalone HTML5 QA Studio in the default web browser.
100% offline, zero runtime server.
"""

from __future__ import annotations

import sys
from la_fat.cli import main_cli

if __name__ == "__main__":
    main_cli(["dashboard"] + sys.argv[1:])
