from __future__ import annotations

from typing import Any, Dict, Tuple

from annotation_tags import parse_tags
from annotation_palette import pick_dominant_color, hex_to_rgb_float

_LEVEL_TO_BOND_WIDTH = {"low": 1.5, "medium": 2.5, "high": 4.0, "max": 6.0}

def build_annotation_styles(meta: Dict[str, Any]) -> Dict[str, Dict[int, Any]]:
    atom_idx_map = meta.get("atom_id_to_idx", {}) or {}
    bond_idx_map = meta.get("bond_key_to_idx", {}) or {}
    atom_comments = meta.get("atoms", {}) or {}
    bond_comments = meta.get("bonds", {}) or {}

    atom_tooltips: Dict[int, str] = {}
    bond_tooltips: Dict[int, str] = {}
    atom_colors: Dict[int, Tuple[float, float, float]] = {}
    bond_colors: Dict[int, Tuple[float, float, float]] = {}
    bond_widths: Dict[int, float] = {}

    for atom_id, comment in atom_comments.items():
        if not comment or atom_id not in atom_idx_map:
            continue
        aidx = atom_idx_map[atom_id]
        atom_tooltips[aidx] = comment

        tags = parse_tags(comment)
        decision = pick_dominant_color(tags)
        if decision:
            atom_colors[aidx] = hex_to_rgb_float(decision.color_hex)

    for bond_key, comment in bond_comments.items():
        if not comment or bond_key not in bond_idx_map:
            continue
        bidx = bond_idx_map[bond_key]
        bond_tooltips[bidx] = comment

        tags = parse_tags(comment)
        decision = pick_dominant_color(tags)
        if decision:
            bond_colors[bidx] = hex_to_rgb_float(decision.color_hex)
            if decision.prop == "break":
                bond_widths[bidx] = 6.0
            elif decision.level in _LEVEL_TO_BOND_WIDTH:
                bond_widths[bidx] = _LEVEL_TO_BOND_WIDTH[decision.level]

    return {
        "atom_tooltips": atom_tooltips,
        "bond_tooltips": bond_tooltips,
        "atom_colors": atom_colors,
        "bond_colors": bond_colors,
        "bond_widths": bond_widths,
    }