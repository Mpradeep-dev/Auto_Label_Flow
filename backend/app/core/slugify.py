"""Tiny dependency-free slugify — good enough for project names, not a
general Unicode-transliteration tool."""
from __future__ import annotations

import re
import uuid

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    slug = _NON_ALNUM.sub("-", text.lower()).strip("-")
    return slug or uuid.uuid4().hex[:8]
