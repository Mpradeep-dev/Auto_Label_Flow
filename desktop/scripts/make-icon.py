#!/usr/bin/env python3
"""Regenerate desktop/icon.ico from the frontend favicon.

The favicon (frontend/public/favicon.png, 256x256 RGBA) is the single source
of truth for the app mark. electron-builder reads desktop/icon.ico for the
Windows app/exe/installer icon (win.icon in electron-builder.yml); main.js
also loads it for the BrowserWindow.

    python scripts/make-icon.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "frontend" / "public" / "favicon.png"
OUT = HERE.parent / "icon.ico"
SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def main() -> None:
    img = Image.open(SRC).convert("RGBA")
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.LANCZOS)
    img.save(OUT, format="ICO", sizes=SIZES)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes) from {SRC}")


if __name__ == "__main__":
    main()
