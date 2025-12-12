"""
Color palettes and priority rules for chemical-property gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

# Palettes: low -> medium -> high -> max
PALETTES: Dict[str, Dict[str, str]] = {
    "acid": {
        "low": "#6EE7B7",
        "medium": "#FDE68A",
        "high": "#F87171",
        "max": "#7F1D1D",
    },
    "base": {
        "low": "#93C5FD",
        "medium": "#3B82F6",
        "high": "#1D4ED8",
        "max": "#4C1D95",
    },
    "nucleophilic": {
        "low": "#BBF7D0",
        "medium": "#22C55E",
        "high": "#166534",
        "max": "#0D9488",
    },
    "electrophilic": {
        "low": "#FEF9C3",
        "medium": "#FACC15",
        "high": "#FB923C",
        "max": "#DC2626",
    },
    "oxidizable": {
        "low": "#E9D5FF",
        "medium": "#A855F7",
        "high": "#6D28D9",
        "max": "#2E1065",
    },
    "reducible": {
        "low": "#E0F2FE",
        "medium": "#38BDF8",
        "high": "#0284C7",
        "max": "#0C4A6E",
    },
    "protect": {
        "low": "#DCFCE7",
        "medium": "#22C55E",
        "high": "#166534",
        "max": "#052E16",
    },
    "break": {"_": "#FF4D4D"},  # make break pop more
}

# Priority: higher wins when multiple tags exist on same atom/bond
PRIORITY: Dict[str, int] = {
    "break": 100,
    "protect": 90,
    "electrophilic": 70,
    "nucleophilic": 70,
    "acid": 50,
    "base": 50,
    "oxidizable": 40,
    "reducible": 40,
}

@dataclass(frozen=True)
class ColorDecision:
    prop: str
    level: str
    color_hex: str
    priority: int

def hex_to_rgb_float(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)

def pick_dominant_color(tags: Dict[str, Any]) -> Optional[ColorDecision]:
    best: Optional[ColorDecision] = None

    for prop, val in tags.items():
        if prop not in PRIORITY:
            continue

        pr = PRIORITY[prop]

        if prop == "break":
            decision = ColorDecision(prop="break", level="_", color_hex=PALETTES["break"]["_"], priority=pr)
        else:
            if not isinstance(val, str):
                continue
            level = val.lower()
            color = PALETTES.get(prop, {}).get(level)
            if not color:
                continue
            decision = ColorDecision(prop=prop, level=level, color_hex=color, priority=pr)

        if best is None or decision.priority > best.priority:
            best = decision

    return best