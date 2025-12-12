"""
可视化 Mermaid 分子图：解析为 RDKit Mol，并在图中标注行内注释。
"""

import re
from typing import Dict, Tuple, Optional, Iterable
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from mermaid_to_rdkit import mermaid_to_mol_with_comments


def visualize_mermaid(mermaid_text: str,
                      size: Tuple[int, int] = (500, 400),
                      save_path: Optional[str] = None,
                      strict: bool = False,
                      respect_h_count: bool = False,
                      show_atom_notes: bool = True,
                      show_bond_notes: bool = True,
                      adjust_atom_labels: bool = True,
                      highlight_atoms: Optional[Iterable[int]] = None,
                      highlight_bonds: Optional[Iterable[int]] = None,
                      highlight_atom_color: Tuple[float, float, float] = (0.25, 0.65, 1.0),
                      highlight_bond_color: Tuple[float, float, float] = (0.6, 0.9, 0.6),
                      highlight_atom_radius: float = 0.35,
                      # NEW
                      atom_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
                      bond_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
                      bond_widths: Optional[Dict[int, float]] = None) -> Optional[str]:
    """
    将 Mermaid 文本转为 RDKit Mol 并输出带注释的 SVG。
    """

    mol, meta = mermaid_to_mol_with_comments(
        mermaid_text,
        strict=strict,
        respect_h_count=respect_h_count
    )
    if mol is None:
        return None

    def _format_label(element: str, h_count: int, charge: int) -> str:
        label = element
        if h_count > 0:
            label += "H" if h_count == 1 else f"H{h_count}"
        if charge != 0:
            sign = "+" if charge > 0 else "-"
            mag = abs(charge)
            suffix = sign if mag == 1 else f"{mag}{sign}"
            label = f"{label}({suffix})"
        return label

    def _parse_raw_label(raw: str) -> Optional[Tuple[str, int, int]]:
        m = re.match(r'^([A-Z][a-z]?)(?:H(\d*))?(?:\((\d*[+-])\))?$', raw.strip())
        if not m:
            return None
        element = m.group(1)
        h_str = m.group(2)
        charge_str = m.group(3)
        if h_str is None:
            h_count = 0
        elif h_str == "":
            h_count = 1
        else:
            h_count = int(h_str)
        charge = 0
        if charge_str:
            sign = charge_str[-1]
            num = charge_str[:-1]
            mag = int(num) if num else 1
            charge = mag if sign == "+" else -mag
        return element, h_count, charge

    def _label_from_atom(atom: Chem.Atom) -> str:
        element = atom.GetSymbol()
        h_count = atom.GetTotalNumHs()
        charge = atom.GetFormalCharge()
        return _format_label(element, h_count, charge)

    mol_for_draw = Chem.Mol(mol)

    for atom_id, comment in meta.get("atoms", {}).items():
        idx = meta.get("atom_id_to_idx", {}).get(atom_id)
        if idx is None or idx < 0 or idx >= mol_for_draw.GetNumAtoms():
            continue
        atom = mol_for_draw.GetAtomWithIdx(idx)
        if adjust_atom_labels:
            raw_label = (meta.get("atom_labels", {}) or {}).get(atom_id)
            raw_parsed = _parse_raw_label(raw_label) if raw_label else None
            if respect_h_count and raw_parsed:
                element, h_count, charge = raw_parsed
                display_label = _format_label(element, h_count, charge)
            else:
                display_label = _label_from_atom(atom)
            atom.SetProp("atomLabel", display_label)
            atom.SetProp("_displayLabel", display_label)
            atom.SetProp("_drawLabel", display_label)
        if comment and show_atom_notes:
            atom.SetProp("atomNote", comment)

    for bond_key, comment in meta.get("bonds", {}).items():
        idx = meta.get("bond_key_to_idx", {}).get(bond_key)
        if idx is not None and 0 <= idx < mol_for_draw.GetNumBonds():
            if comment and show_bond_notes:
                mol_for_draw.GetBondWithIdx(idx).SetProp("bondNote", comment)

    rdMolDraw2D.PrepareMolForDrawing(mol_for_draw, addChiralHs=False)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.fillHighlights = True

    atom_colors = atom_colors or {}
    bond_colors = bond_colors or {}

    highlight_atoms_set = set(highlight_atoms or []) | set(atom_colors.keys())
    highlight_bonds_set = set(highlight_bonds or []) | set(bond_colors.keys())
    highlight_atoms_list = sorted(highlight_atoms_set)
    highlight_bonds_list = sorted(highlight_bonds_set)

    highlight_atom_colors: Dict[int, Tuple[float, float, float]] = {
        idx: atom_colors.get(idx, highlight_atom_color) for idx in highlight_atoms_list
    }
    highlight_bond_colors: Dict[int, Tuple[float, float, float]] = {
        idx: bond_colors.get(idx, highlight_bond_color) for idx in highlight_bonds_list
    }
    highlight_atom_radii = {idx: highlight_atom_radius for idx in highlight_atoms_list}

    # Approximate bond width via global multiplier (RDKit public API limitation)
    if bond_widths:
        max_w = max(bond_widths.values()) if bond_widths.values() else 1.0
        opts.highlightBondWidthMultiplier = max(1.0, min(3.0, max_w / 2.0))

    drawer.DrawMolecule(
        mol_for_draw,
        highlightAtoms=highlight_atoms_list,
        highlightBonds=highlight_bonds_list,
        highlightAtomColors=highlight_atom_colors,
        highlightBondColors=highlight_bond_colors,
        highlightAtomRadii=highlight_atom_radii,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(svg)

    return svg