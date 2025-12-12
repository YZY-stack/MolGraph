"""
Parse inline Mermaid comments like:
    %% [acid=high] [nucleophilic=medium] free text
    %% [break]
into structured tags dict.

- key: str
- value: 'low'|'medium'|'high'|'max' or bool for flag tags (e.g. break)
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Union

TagValue = Union[str, bool]

_TAG_RE = re.compile(r"\[([a-zA-Z_][a-zA-Z0-9_-]*)(?:=([a-zA-Z0-9_-]+))?\]")

_ALLOWED_LEVELS = {"low", "medium", "high", "max"}

def parse_tags(comment: Optional[str]) -> Dict[str, TagValue]:
    if not comment:
        return {}

    tags: Dict[str, TagValue] = {}
    for m in _TAG_RE.finditer(comment):
        key = m.group(1).strip()
        raw_val = m.group(2)
        if raw_val is None:
            tags[key] = True
        else:
            v = raw_val.strip().lower()
            tags[key] = v if v in _ALLOWED_LEVELS else raw_val.strip()
    return tags