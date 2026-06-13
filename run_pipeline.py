#!/usr/bin/env python3
"""LA Fat Segmentation — CLI entry point.

Usage::

    python run_pipeline.py --patient 0674
    python run_pipeline.py --patient 0674 --config config.yaml
    python run_pipeline.py --patient 0674 --data-dir data --output-dir outputs

This is a thin wrapper around :func:`la_fat.pipeline.main_cli`.
"""

from la_fat.pipeline import main_cli

if __name__ == "__main__":
    main_cli()
