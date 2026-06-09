#!/usr/bin/env python3
"""Compatibility hook entrypoint for binary text classification project blocks."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    hook_path = Path(__file__).with_name("check_binary_project.py")
    runpy.run_path(str(hook_path), run_name="__main__")
