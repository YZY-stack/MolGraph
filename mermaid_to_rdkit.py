"""
将Mermaid Graph格式转换为RDKit Mol对象

RDKit分子表示方式：
1. Mol对象 - 分子的主要容器
2. Atom对象 - 包含原子序号、元素符号、氢数、形式电荷等
3. Bond对象 - 包含起始/终止原子索引、键类型（SINGLE, DOUBLE, TRIPLE等）
4. 邻接表结构 - 通过原子索引连接

示例：
    乙醇 (CH3CH2OH)
    - 3个原子: C(idx=0), C(idx=1), O(idx=2)
    - 2个键: C-C (0-1, SINGLE), C-O (1-2, SINGLE)
    - 隐式H会自动计算
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from exceptiongroup import print_exc
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


@dataclass
class GraphEntry:
    """Single Mermaid line representation."""

    kind: str
    line_no: int
    text: str
    indent: str = ""
    subgraph: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubgraphInfo:
    subgraph_id: str
    label: str
    parent_id: Optional[str]
    start_idx: int
    end_idx: int


@dataclass
class ParsedGraph:
    direction: str
    notes: Optional[str]
    entries: List[Dict[str, Any]]
    atoms: Dict[str, Any]
    bonds: List[Tuple[str, ...]]
    subgraphs: Dict[str, SubgraphInfo]
    atom_comments: Dict[str, str] = field(default_factory=dict)
    bond_comments: Dict[Tuple[str, ...], str] = field(default_factory=dict)


@dataclass
class MolMapping:
    atom_id_to_idx: Dict[str, int] = field(default_factory=dict)
    bond_key_to_idx: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    entry_to_atom_idx: Dict[int, int] = field(default_factory=dict)
    entry_to_bond_idx: Dict[int, int] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


class MermaidMolParser:
    """解析Mermaid分子图并转换为RDKit Mol对象"""

    # 键类型映射
    BOND_TYPE_MAP = {
        '---': Chem.BondType.SINGLE,
        '===': Chem.BondType.DOUBLE,
        '-.-': Chem.BondType.TRIPLE,
        '-->': Chem.BondType.DATIVE,  # 配位键
    }

    def __init__(self, respect_h_count: bool = False):
        # respect_h_count=True 时使用标签中的显式H数，False时仅依据骨架构建
        self.respect_h_count = respect_h_count
        self.atoms: Dict[str, str] = {}  # atom_id -> atom_label映射
        self.bonds: List[Tuple] = []  # (atom1_id, atom2_id, bond_type) 或 (atom1_id, atom2_id, bond_type, stereo)
        self.chirality: Dict[str, str] = {}  # atom_id -> chirality ('R' or 'S') 映射
        # 保存行内注释，用于可视化或调试
        self.atom_comments: Dict[str, str] = {}  # atom_id -> comment
        self.bond_comments: Dict[Tuple[str, ...], str] = {}  # (atom1_id, atom2_id, bond_type[, stereo]) -> comment
        # 保存索引映射，便于从Mol回溯到Mermaid ID
        self.atom_id_to_idx: Dict[str, int] = {}  # atom_id -> atom_idx
        self.bond_key_to_idx: Dict[Tuple[str, ...], int] = {}  # bond_key -> bond_idx
        # 最近一次解析的结构化信息
        self.last_parsed_graph: Optional[ParsedGraph] = None
        self.last_mapping: Optional[MolMapping] = None
        self.last_warnings: List[str] = []

    def parse_mermaid_graph(self, mermaid_text: str) -> Optional[Chem.Mol]:
        """
        解析Mermaid图文本并生成RDKit Mol对象

        Args:
            mermaid_text: Mermaid格式的分子图文本

        Returns:
            RDKit Mol对象，解析失败返回None
        """
        self.atoms = {}
        self.bonds = []
        self.chirality = {}
        self.atom_comments = {}
        self.bond_comments = {}
        self.atom_id_to_idx = {}
        self.bond_key_to_idx = {}

        lines = mermaid_text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # 跳过注释、空行、graph声明、subgraph声明
            if (line.startswith('%%') or
                line.startswith('graph ') or
                line.startswith('subgraph ') or
                line == 'end' or
                not line):
                continue

            # 提取行内注释（形如 ... %% comment），并返回去掉注释的代码行
            line, inline_comment = self._strip_inline_comment(line)
            if not line:
                # 只剩注释内容，跳过
                continue

            # 解析原子定义和键连接
            self._parse_line(line, inline_comment)

        # 构建RDKit Mol对象
        return self._build_mol()

    def _strip_inline_comment(self, line: str) -> Tuple[str, Optional[str]]:
        """
        拆分行内注释并返回纯代码部分

        返回:
            code_part: 去掉注释后的代码
            comment_part: 行内注释（如果存在）
        """
        if '%%' not in line:
            return line, None

        code_part, comment_part = line.split('%%', 1)
        return code_part.rstrip(), comment_part.strip() or None

    def _record_atom_comment(self, atom_id: str, comment: Optional[str]):
        """记录与原子相关的行内注释"""
        if comment and atom_id not in self.atom_comments:
            self.atom_comments[atom_id] = comment

    def _record_bond_comment(self, key: Tuple[str, ...], comment: Optional[str]):
        """记录与键相关的行内注释"""
        if comment and key not in self.bond_comments:
            self.bond_comments[key] = comment

    def get_inline_comments(self) -> Dict[str, Dict[str, str]]:
        """
        获取解析过程中收集的行内注释

        Returns:
            {'atoms': {atom_id: comment}, 'bonds': {(atom1, atom2, bond_type[, stereo]): comment}}
        """
        return {
            'atoms': self.atom_comments,
            'bonds': self.bond_comments
        }

    def get_atom_labels(self) -> Dict[str, str]:
        """
        获取原子ID到原始标签的映射（如 CH3、O(-)），用于可视化显示。
        """
        return self.atoms

    def get_index_mappings(self) -> Dict[str, Dict]:
        """
        获取Mermaid ID 与 RDKit 索引的映射

        Returns:
            {'atom_id_to_idx': {...}, 'bond_key_to_idx': {...}}
        """
        return {
            'atom_id_to_idx': self.atom_id_to_idx,
            'bond_key_to_idx': self.bond_key_to_idx
        }

    def parse_mermaid_with_structure(
        self, mermaid_text: str, strict: bool = False
    ) -> Tuple[Optional[Chem.Mol], ParsedGraph, MolMapping, List[str]]:
        """解析Mermaid文本，返回Mol、行级结构、映射与警告。"""
        warnings: List[str] = []
        entries: List[Dict[str, object]] = []
        subgraphs: Dict[str, SubgraphInfo] = {}
        subgraph_stack: List[Tuple[str, str, int]] = []  # (id, label, entry_idx)

        direction = "TB"
        notes: Optional[str] = None

        # reset parser state
        self.atoms = {}
        self.bonds = []
        self.chirality = {}
        self.atom_comments = {}
        self.bond_comments = {}
        self.atom_id_to_idx = {}
        self.bond_key_to_idx = {}

        for line_no, raw_line in enumerate(mermaid_text.splitlines(), start=1):
            indent = raw_line[: len(raw_line) - len(raw_line.lstrip(" "))]
            line = raw_line.strip()
            current_sub = subgraph_stack[-1][0] if subgraph_stack else None

            if line == "":
                entries.append({"type": "blank", "line_no": line_no, "text": raw_line, "indent": indent, "subgraph": current_sub})
                continue

            if line.startswith("graph "):
                parts = line.split()
                if len(parts) >= 2:
                    direction = parts[1]
                else:
                    warnings.append(f"Line {line_no}: graph directive missing direction, defaulting to TB")
                entries.append({"type": "graph", "line_no": line_no, "text": raw_line, "indent": indent, "subgraph": None, "direction": direction})
                continue

            if line.startswith("subgraph "):
                match = re.match(r'subgraph\s+([^\s\[]+)(?:\s*\["?(.*?)"?\])?', line)
                if match:
                    sub_id = match.group(1)
                    label = match.group(2).strip('"') if match.group(2) else sub_id
                    subgraph_stack.append((sub_id, label, len(entries)))
                    entries.append(
                        {
                            "type": "subgraph",
                            "line_no": line_no,
                            "text": raw_line,
                            "indent": indent,
                            "subgraph": current_sub,
                            "subgraph_id": sub_id,
                            "label": label,
                        }
                    )
                else:
                    warnings.append(f"Line {line_no}: invalid subgraph syntax '{line}'")
                    entries.append({"type": "other", "line_no": line_no, "text": raw_line, "indent": indent, "subgraph": current_sub})
                continue

            if line == "end":
                if subgraph_stack:
                    sub_id, label, start_idx = subgraph_stack.pop()
                    entries.append({"type": "end", "line_no": line_no, "text": raw_line, "indent": indent, "subgraph": current_sub})
                    subgraphs[sub_id] = SubgraphInfo(
                        subgraph_id=sub_id,
                        label=label,
                        parent_id=subgraph_stack[-1][0] if subgraph_stack else None,
                        start_idx=start_idx,
                        end_idx=len(entries) - 1,
                    )
                else:
                    warnings.append(f"Line {line_no}: stray end without matching subgraph")
                    entries.append({"type": "other", "line_no": line_no, "text": raw_line, "indent": indent, "subgraph": current_sub})
                continue

            if line.startswith("%%"):
                if "Notes:" in line and notes is None:
                    notes = line.lstrip("%").strip().replace("Notes:", "", 1).strip()
                entries.append({"type": "comment", "line_no": line_no, "text": raw_line, "indent": indent, "subgraph": current_sub})
                continue

            code_part, inline_comment = self._strip_inline_comment(line)
            success = self._parse_line(
                code_part,
                inline_comment,
                entries=entries,
                current_subgraph=current_sub,
                line_no=line_no,
                indent=indent,
                warnings=warnings,
                raw_text=raw_line,
            )
            if not success:
                entries.append({"type": "other", "line_no": line_no, "text": raw_line, "indent": indent, "subgraph": current_sub})
                warnings.append(f"Line {line_no}: unrecognized content '{line}'")

        if subgraph_stack:
            warnings.append("Unclosed subgraph block(s) detected")

        for bond in self.bonds:
            a1, a2 = bond[0], bond[1]
            if a1 not in self.atoms:
                warnings.append(f"Bond references missing atom(s): {a1}")
            elif a2 not in self.atoms:
                warnings.append(f"Bond references missing atom(s): {a2}")

        parsed_graph = ParsedGraph(
            direction=direction,
            notes=notes,
            entries=entries.copy(),
            atoms=self.atoms.copy(),
            bonds=self.bonds.copy(),
            subgraphs=subgraphs.copy(),
            atom_comments=self.atom_comments.copy(),
            bond_comments=self.bond_comments.copy(),
        )

        mapping = MolMapping()
        mol = self._build_mol()

        if mol is None and strict:
            mapping.issues.append("RDKit molecule could not be constructed")
        elif mol is None:
            warnings.append("RDKit molecule could not be constructed")

        idx_map = self.get_index_mappings()
        mapping.atom_id_to_idx = idx_map.get("atom_id_to_idx", {})
        mapping.bond_key_to_idx = idx_map.get("bond_key_to_idx", {})

        for idx, entry in enumerate(entries):
            if entry.get("type") == "atom":
                atom_id = entry.get("atom_id")
                if atom_id in mapping.atom_id_to_idx:
                    mapping.entry_to_atom_idx[idx] = mapping.atom_id_to_idx[atom_id]
                else:
                    warnings.append(f"Line {entry.get('line_no')}: atom id '{atom_id}' not found in RDKit mapping")
            if entry.get("type") == "bond":
                atom1 = entry.get("atom1")
                atom2 = entry.get("atom2")
                bond_type = entry.get("bond_type")
                stereo = entry.get("stereo")
                key = (atom1, atom2, bond_type, stereo) if stereo else (atom1, atom2, bond_type)
                bond_idx = mapping.bond_key_to_idx.get(key)
                if bond_idx is None and bond_type != "-->":
                    rev_key = (atom2, atom1, bond_type, stereo) if stereo else (atom2, atom1, bond_type)
                    bond_idx = mapping.bond_key_to_idx.get(rev_key)
                if bond_idx is not None:
                    mapping.entry_to_bond_idx[idx] = bond_idx
                else:
                    warnings.append(f"Line {entry.get('line_no')}: bond not mapped to RDKit object")

        # cache last parse
        self.last_parsed_graph = parsed_graph
        self.last_mapping = mapping
        self.last_warnings = list(warnings)

        if mol is not None:
            rev_atom = {idx: atom_id for atom_id, idx in mapping.atom_id_to_idx.items()}
            rev_bond = {idx: key for key, idx in mapping.bond_key_to_idx.items()}
            atom_idx_info: Dict[int, Dict[str, object]] = {}
            for entry_idx, aidx in mapping.entry_to_atom_idx.items():
                entry = entries[entry_idx] if 0 <= entry_idx < len(entries) else {}
                atom_idx_info[aidx] = {"line_no": entry.get("line_no"), "text": entry.get("text")}

            sanitize_msgs = _sanitize_with_warnings(
                mol,
                atom_idx_to_id=rev_atom,
                bond_idx_to_key=rev_bond,
                atom_idx_info=atom_idx_info,
            )
            if sanitize_msgs:
                warnings.extend(sanitize_msgs)
                if strict:
                    mapping.issues.extend(sanitize_msgs)

        return mol, parsed_graph, mapping, warnings

    def _parse_line(
        self,
        line: str,
        inline_comment: Optional[str],
        entries: Optional[List[Dict[str, object]]] = None,
        current_subgraph: Optional[str] = None,
        line_no: Optional[int] = None,
        indent: str = "",
        warnings: Optional[List[str]] = None,
        raw_text: Optional[str] = None,
    ) -> bool:
        """解析单行，提取原子和键信息，可选记录 entry。"""

        stereo_bond_pattern = r'([\w_]+)\s*===\|([EZez]|cis|trans|CIS|TRANS)\|\s*([\w_]+)'
        stereo_match = re.search(stereo_bond_pattern, line)

        if stereo_match:
            atom1_id = stereo_match.group(1)
            stereo_type = stereo_match.group(2).upper()
            atom2_id = stereo_match.group(3)
            self.bonds.append((atom1_id, atom2_id, '===', stereo_type))
            self._record_bond_comment((atom1_id, atom2_id, '===', stereo_type), inline_comment)
            if entries is not None:
                entries.append(
                    {
                        "type": "bond",
                        "line_no": line_no,
                        "text": raw_text if raw_text is not None else line,
                        "indent": indent,
                        "subgraph": current_subgraph,
                        "atom1": atom1_id,
                        "atom2": atom2_id,
                        "bond_type": "===",
                        "stereo": stereo_type,
                        "comment": inline_comment,
                    }
                )
            return True

        bond_pattern = r'([\w_]+)\s*(---|\===|-\.-|-->)\s*([\w_]+)'
        bond_match = re.search(bond_pattern, line)

        if bond_match:
            atom1_id = bond_match.group(1)
            bond_type = bond_match.group(2)
            atom2_id = bond_match.group(3)
            self.bonds.append((atom1_id, atom2_id, bond_type))
            self._record_bond_comment((atom1_id, atom2_id, bond_type), inline_comment)
            if entries is not None:
                entries.append(
                    {
                        "type": "bond",
                        "line_no": line_no,
                        "text": raw_text if raw_text is not None else line,
                        "indent": indent,
                        "subgraph": current_subgraph,
                        "atom1": atom1_id,
                        "atom2": atom2_id,
                        "bond_type": bond_type,
                        "stereo": None,
                        "comment": inline_comment,
                    }
                )
            return True

        atom_pattern = r'([\w_]+?)(?:_(R|S))?\[([^\]]+)\]'
        atom_match = re.search(atom_pattern, line)

        if atom_match:
            base_id = atom_match.group(1)
            chirality = atom_match.group(2)
            label = atom_match.group(3)

            if chirality:
                atom_id = f"{base_id}_{chirality}"
                self.chirality[atom_id] = chirality
            else:
                atom_id = base_id

            if atom_id in self.atoms:
                if warnings is not None:
                    warnings.append(f"Line {line_no}: duplicate atom id '{atom_id}' skipped")
                return True

            self.atoms[atom_id] = label
            self._record_atom_comment(atom_id, inline_comment)
            if entries is not None:
                entries.append(
                    {
                        "type": "atom",
                        "line_no": line_no,
                        "text": raw_text if raw_text is not None else line,
                        "indent": indent,
                        "subgraph": current_subgraph,
                        "atom_id": atom_id,
                        "label": label,
                        "comment": inline_comment,
                    }
                )
            return True

        return False

    def _parse_atom_label(self, label: str) -> Tuple[str, int, int]:
        """
        解析原子标签，提取元素符号、显式氢数和电荷

        Args:
            label: 原子标签，如 'C', 'OH', 'NH2', 'N(+)', 'O(-)', 'O(2-)'

        Returns:
            (元素符号, 显式氢数, 形式电荷)
            无效标签返回 ('*', 0, 0) - 使用Dummy Atom标记
        """
        label = label.strip()

        # 匹配元素符号、氢数和电荷（新格式：括号包裹，数字在前，符号在后）
        # 例如: C, OH, NH2, N(+), NH2(+), O(-), O(2-)
        match = re.match(r'^([A-Z][a-z]?)(?:H(\d*))?(?:\((\d*[+-])\))?$', label)

        if match:
            element = match.group(1)
            h_count_str = match.group(2)
            charge_str = match.group(3)

            # 验证是否为合法元素符号
            try:
                # 尝试创建原子以验证元素符号合法性
                test_atom = Chem.Atom(element)
                atomic_num = test_atom.GetAtomicNum()

                # 检查是否为真实元素（原子序数>0）
                if atomic_num == 0 and element != '*':
                    # 不是合法元素，返回Dummy Atom
                    return '*', 0, 0

            except Exception:
                # 创建失败，返回Dummy Atom
                return '*', 0, 0

            # 解析氢数
            if h_count_str is None:
                # 没有H标记
                h_count = 0
            elif h_count_str == '':
                # 有H但没有数字，表示1个H
                h_count = 1
            else:
                # 明确指定H的数量
                h_count = int(h_count_str)

            # 解析电荷（新格式：数字在前，符号在后）
            charge = 0
            if charge_str:
                # charge_str 格式: "+", "-", "2+", "2-", "3+" 等
                if charge_str == '+':
                    charge = 1
                elif charge_str == '-':
                    charge = -1
                else:
                    # 提取数字和符号
                    sign = charge_str[-1]  # 最后一个字符是符号
                    number = charge_str[:-1]  # 前面的是数字
                    magnitude = int(number)
                    charge = magnitude if sign == '+' else -magnitude

            # respect_h_count=False 时忽略标签里的显式氢数，由RDKit按骨架推断
            if not self.respect_h_count:
                h_count = 0

            return element, h_count, charge

        # 如果无法解析，返回Dummy Atom标记
        # Dummy Atom: 原子序数为0，符号为'*'，不会与真实元素混淆
        return '*', 0, 0

    def _build_mol(self) -> Optional[Chem.Mol]:
        """根据解析的原子和键信息构建RDKit Mol对象"""

        if not self.atoms:
            return None

        # 创建可编辑的分子
        mol = Chem.RWMol()

        # 原子ID到索引的映射
        atom_id_to_idx = {}

        # 添加原子
        for atom_id, label in self.atoms.items():
            element, h_count, charge = self._parse_atom_label(label)

            # 创建原子
            atom = Chem.Atom(element)

            # 设置显式氢数（如果有）
            if h_count > 0:
                atom.SetNumExplicitHs(h_count)

            # 设置形式电荷（如果有）
            if charge != 0:
                atom.SetFormalCharge(charge)

            # 添加到分子中
            idx = mol.AddAtom(atom)
            atom_id_to_idx[atom_id] = idx
            self.atom_id_to_idx[atom_id] = idx

            # 设置手性（如果有）
            if atom_id in self.chirality:
                chirality_type = self.chirality[atom_id]
                atom_obj = mol.GetAtomWithIdx(idx)

                if chirality_type == 'R':
                    atom_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                elif chirality_type == 'S':
                    atom_obj.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)

        # 添加键
        for bond_info in self.bonds:
            if len(bond_info) == 3:
                # 普通键: (atom1_id, atom2_id, bond_type_str)
                atom1_id, atom2_id, bond_type_str = bond_info
                stereo_type = None
            elif len(bond_info) == 4:
                # 带立体化学的键: (atom1_id, atom2_id, bond_type_str, stereo_type)
                atom1_id, atom2_id, bond_type_str, stereo_type = bond_info
            else:
                continue

            if atom1_id in atom_id_to_idx and atom2_id in atom_id_to_idx:
                idx1 = atom_id_to_idx[atom1_id]
                idx2 = atom_id_to_idx[atom2_id]
                bond_type = self.BOND_TYPE_MAP.get(bond_type_str, Chem.BondType.SINGLE)

                mol.AddBond(idx1, idx2, bond_type)
                # 记录键索引映射（用于注释回溯）
                bond_obj = mol.GetBondBetweenAtoms(idx1, idx2)
                if bond_obj:
                    bond_idx = bond_obj.GetIdx()
                    bond_key = (atom1_id, atom2_id, bond_type_str) if stereo_type is None else (atom1_id, atom2_id, bond_type_str, stereo_type)
                    self.bond_key_to_idx[bond_key] = bond_idx

                # 设置立体化学（稍后统一处理，需要先添加所有键）
                if stereo_type:
                    # 记录需要设置立体化学的键
                    if not hasattr(mol, '_stereo_bonds'):
                        mol._stereo_bonds = []
                    mol._stereo_bonds.append((idx1, idx2, stereo_type))

        # 在转换为不可编辑的Mol之前，设置立体化学
        if hasattr(mol, '_stereo_bonds'):
            for idx1, idx2, stereo_type in mol._stereo_bonds:
                bond = mol.GetBondBetweenAtoms(idx1, idx2)

                # 获取双键两端原子的邻接原子（用于定义立体化学）
                atom1 = mol.GetAtomWithIdx(idx1)
                atom2 = mol.GetAtomWithIdx(idx2)

                # 找到idx1的邻居（除了idx2）
                neighbors1 = [n.GetIdx() for n in atom1.GetNeighbors() if n.GetIdx() != idx2]
                # 找到idx2的邻居（除了idx1）
                neighbors2 = [n.GetIdx() for n in atom2.GetNeighbors() if n.GetIdx() != idx1]

                # 如果两端都有邻居，设置立体化学
                if neighbors1 and neighbors2:
                    # 使用第一个邻居作为参考原子
                    bond.SetStereoAtoms(neighbors1[0], neighbors2[0])

                    if stereo_type == 'E':
                        bond.SetStereo(Chem.BondStereo.STEREOE)
                    elif stereo_type == 'Z':
                        bond.SetStereo(Chem.BondStereo.STEREOZ)
                    elif stereo_type == 'CIS':
                        bond.SetStereo(Chem.BondStereo.STEREOCIS)
                    elif stereo_type == 'TRANS':
                        bond.SetStereo(Chem.BondStereo.STEREOTRANS)

        # 转换为不可编辑的Mol对象
        mol = mol.GetMol()

        return mol


def _sanitize_with_warnings(
    mol: Chem.Mol,
    atom_idx_to_id: Optional[Dict[int, str]] = None,
    bond_idx_to_key: Optional[Dict[int, Tuple[str, ...]]] = None,
    atom_idx_info: Optional[Dict[int, Dict[str, object]]] = None,
) -> List[str]:
    """Run RDKit sanitization and collect warnings with Mermaid context when possible."""
    messages: List[str] = []
    # Map sanitize flags to readable names
    flag_names = {
        int(getattr(Chem.SanitizeFlags, name)): name
        for name in dir(Chem.SanitizeFlags)
        if name.startswith("SANITIZE_")
    }

    try:
        fail_flags = Chem.SanitizeMol(mol, catchErrors=True)
        if fail_flags != Chem.SanitizeFlags.SANITIZE_NONE:
            for flag in Chem.SanitizeFlags.values:
                if int(fail_flags) & int(flag):
                    flag_name = flag_names.get(int(flag), str(flag))
                    # messages.append(f"Sanitize failed: {flag_name} (flag {int(flag)})")
    except Exception as exc:
        messages.append(f"Sanitize exception: {exc}")

    try:
        for prob in Chem.DetectChemistryProblems(mol) or []:
            msg = f"Chemistry problem: {prob.GetType()}"
            if prob.GetType() == "AtomValenceException":
                aidx = prob.GetAtomIdx()
                mermaid_id = atom_idx_to_id.get(aidx) if atom_idx_to_id else None
                # msg += f" at atom idx {aidx}"
                if mermaid_id:
                    msg += f" (Mermaid id: {mermaid_id})"
                if atom_idx_info and aidx in atom_idx_info:
                    info = atom_idx_info[aidx]
                    line_no = info.get("line_no")
                    line_text = info.get("text") or ""
                    if line_no:
                        msg += f" [line {line_no}: {line_text.strip()}]"
            elif prob.GetType() == "KekulizeException":
                atom_indices = prob.GetAtomIndices()
                for aidx in atom_indices:
                    mermaid_id = atom_idx_to_id.get(aidx) if atom_idx_to_id else None
                    # msg += f" at atom idx {aidx}"
                    if mermaid_id:
                        msg += f" (Mermaid id: {mermaid_id})"
                    if atom_idx_info and aidx in atom_idx_info:
                        info = atom_idx_info[aidx]
                        line_no = info.get("line_no")
                        line_text = info.get("text") or ""
                        if line_no:
                            msg += f" [line {line_no}: {line_text.strip()}]"
            # if prob.Message():
            #     msg += f" - {prob.Message()}"
            messages.append(msg)
    except Exception as exc:
        print(exc)

    # If sanitize failed but we have no detailed issues, try to surface the exception message with atom mapping
    if messages and not any("Chemistry problem" in m for m in messages):
        try:
            Chem.SanitizeMol(Chem.Mol(mol))  # may raise with detailed text
        except Exception as exc:
            emsg = str(exc)
            atom_match = re.search(r"atom # (\\d+)", emsg)
            if atom_match:
                idx = int(atom_match.group(1))
                mermaid_id = atom_idx_to_id.get(idx) if atom_idx_to_id else None
                if mermaid_id:
                    emsg = f"{emsg} (Mermaid id: {mermaid_id})"
                if atom_idx_info and idx in atom_idx_info:
                    info = atom_idx_info[idx]
                    line_no = info.get("line_no")
                    line_text = info.get("text") or ""
                    if line_no:
                        emsg = f"{emsg} [line {line_no}: {line_text.strip()}]"
            messages.append(f"Sanitize detail: {emsg}")

    return messages


def has_invalid_atoms(mol: Chem.Mol) -> bool:
    """
    检查分子是否包含无效原子（Dummy Atom）

    Args:
        mol: RDKit Mol对象

    Returns:
        True表示包含无效原子
    """
    if mol is None:
        return False

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # 原子序数0表示Dummy Atom
            return True
    return False


def get_invalid_atom_labels(mermaid_text: str, mol: Chem.Mol) -> List[str]:
    """
    获取所有无效原子的原始标签

    Args:
        mermaid_text: 原始Mermaid文本
        mol: 解析后的Mol对象

    Returns:
        无效原子的标签列表
    """
    if mol is None:
        return []

    invalid_labels = []
    parser = MermaidMolParser()

    # 重新解析以获取原始标签
    lines = mermaid_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        atom_pattern = r'([\w_]+?)(?:_(R|S))?\[([^\]]+)\]'
        match = re.search(atom_pattern, line)

        if match:
            label = match.group(3)
            element, _, _ = parser._parse_atom_label(label)
            if element == '*':
                invalid_labels.append(label)

    return invalid_labels


def mermaid_to_mol(mermaid_text: str,
                   strict: bool = True,
                   respect_h_count: bool = False) -> Optional[Chem.Mol]:
    """
    将Mermaid图格式转换为RDKit Mol对象（便捷函数）

    Args:
        mermaid_text: Mermaid格式的分子图文本
        strict: 严格模式。如果为True，遇到无效原子时返回None；
                如果为False，使用Dummy Atom (*)标记无效原子并继续
        respect_h_count: 是否使用标签中的显式H数量（True 保留标签的H计数，False按骨架推断）

    Returns:
        RDKit Mol对象，解析失败返回None

    Example:
        >>> mermaid_graph = '''
        ... graph TB
        ...     subgraph Ethanol["Ethanol"]
        ...         Ethanol_C_1[C]
        ...         Ethanol_C_2[C]
        ...         Ethanol_O_3[OH]
        ...
        ...         Ethanol_C_1 --- Ethanol_C_2
        ...         Ethanol_C_2 --- Ethanol_O_3
        ...     end
        ... '''
        >>> mol = mermaid_to_mol(mermaid_graph)
        >>> print(Chem.MolToSmiles(mol))
        CCO
    """
    parser = MermaidMolParser(respect_h_count=respect_h_count)
    mol = parser.parse_mermaid_graph(mermaid_text)

    if mol is None:
        return None

    # 检查是否包含无效原子
    if strict and has_invalid_atoms(mol):
        invalid_labels = get_invalid_atom_labels(mermaid_text, mol)
        print(f"警告: 检测到无效原子标签: {invalid_labels}")
        print(f"这些原子已被标记为Dummy Atom (符号='*', 原子序数=0)")
        print(f"提示: 使用 strict=False 参数可以允许Dummy Atom")
        return None

    return mol


def mermaid_to_mol_with_comments(mermaid_text: str,
                                 strict: bool = True,
                                 respect_h_count: bool = False) -> Tuple[Optional[Chem.Mol], Dict[str, Dict[str, str]]]:
    """
    与 mermaid_to_mol 类似，但同时返回行内注释映射（原子/键）

    Args:
        mermaid_text: Mermaid格式的分子图文本
        strict: 严格模式，遇到无效原子时返回None
        respect_h_count: 是否使用标签中的显式H数量（True 保留标签的H计数，False按骨架推断）

    Returns:
        (
            mol,
            {
                'atoms': {atom_id: comment},
                'bonds': {(atom1, atom2, bond_type[, stereo]): comment},
                'atom_id_to_idx': {atom_id: atom_idx},
                'bond_key_to_idx': {(atom1, atom2, bond_type[, stereo]): bond_idx}
            }
        )
    """
    parser = MermaidMolParser(respect_h_count=respect_h_count)
    mol = parser.parse_mermaid_graph(mermaid_text)

    if mol is None:
        meta = parser.get_inline_comments()
        meta.update(parser.get_index_mappings())
        meta["atom_labels"] = parser.get_atom_labels()
        return None, meta

    if strict and has_invalid_atoms(mol):
        invalid_labels = get_invalid_atom_labels(mermaid_text, mol)
        print(f"警告: 检测到无效原子标签: {invalid_labels}")
        print(f"这些原子已被标记为Dummy Atom (符号='*', 原子序数=0)")
        print(f"提示: 使用 strict=False 参数可以允许Dummy Atom")
        meta = parser.get_inline_comments()
        meta.update(parser.get_index_mappings())
        meta["atom_labels"] = parser.get_atom_labels()
        return None, meta

    meta = parser.get_inline_comments()
    meta.update(parser.get_index_mappings())
    meta["atom_labels"] = parser.get_atom_labels()
    return mol, meta


def parse_mermaid_full(mermaid_text: str,
                       strict: bool = False,
                       respect_h_count: bool = False) -> Tuple[Optional[Chem.Mol], ParsedGraph, MolMapping, List[str]]:
    """
    Parse Mermaid with per-line structure, subgraph info, and RDKit mapping.

    Returns:
        (mol, parsed_graph, mapping, warnings)
    """
    parser = MermaidMolParser(respect_h_count=respect_h_count)
    return parser.parse_mermaid_with_structure(mermaid_text, strict=strict)


def mol_to_smiles(mol: Chem.Mol) -> str:
    """将RDKit Mol转换为SMILES字符串"""
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def mol_to_inchi(mol: Chem.Mol) -> str:
    """将RDKit Mol转换为InChI字符串"""
    if mol is None:
        return ""
    return Chem.MolToInchi(mol)


def visualize_mol(mol: Chem.Mol, title: str = "", save_path: str = None):
    """
    可视化单个分子结构

    Args:
        mol: RDKit Mol对象
        title: 图像标题
        save_path: 保存路径，如果为None则不保存
    """
    if mol is None:
        print(f"无法可视化 {title}: Mol对象为None")
        return

    img = Draw.MolToImage(mol, size=(300, 300))

    if save_path:
        img.save(save_path)
        print(f"已保存到: {save_path}")

    return img


def visualize_mols_grid(mols: List[Chem.Mol],
                        legends: List[str] = None,
                        mols_per_row: int = 4,
                        sub_img_size: Tuple[int, int] = (300, 300),
                        save_path: str = None):
    """
    以网格形式可视化多个分子

    Args:
        mols: RDKit Mol对象列表
        legends: 每个分子的标签列表
        mols_per_row: 每行显示的分子数
        sub_img_size: 每个分子图像的大小
        save_path: 保存路径，如果为None则不保存

    Returns:
        PIL Image对象
    """
    if not mols:
        print("没有分子可以可视化")
        return None

    # 过滤掉None值
    valid_data = [(mol, legends[i] if legends else f"Mol {i+1}")
                  for i, mol in enumerate(mols) if mol is not None]

    if not valid_data:
        print("所有分子都是None")
        return None

    valid_mols, valid_legends = zip(*valid_data)

    img = Draw.MolsToGridImage(
        valid_mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=valid_legends
    )

    if save_path:
        img.save(save_path)
        print(f"网格图已保存到: {save_path}")

    return img


if __name__ == "__main__":
    # 测试示例
    print("=" * 60)
    print("Mermaid分子图 -> RDKit Mol 转换测试")
    print("=" * 60)

    # 定义测试分子
    test_molecules = [
        ("mol1", """
        graph TB
            %% 原始分子名称: Mol1
            subgraph Mol1["Mol1"]
                Mol1_O_1[O]
                Mol1_C_1[C]
                Mol1_C_2[C]
                Mol1_O_2[OH]
                Mol1_C_3[C]
                Mol1_C_4[C]
                Mol1_C_5[C]
                Mol1_C_6[C]
                Mol1_C_7[CH]
                Mol1_O_3[O]
                Mol1_C_8[C]
                Mol1_C_9[C]
                Mol1_C_10[CH2]
                Mol1_C_11[CH]
                Mol1_C_12[CH]
                Mol1_C_13[CH]
                Mol1_C_14[CH]
                Mol1_C_15[C]
                Mol1_O_4[OH]
                Mol1_C_16[CH]
                Mol1_C_17[C]
                Mol1_O_5[OH]
                Mol1_C_18[CH]
                Mol1_O_6[O]
                Mol1_C_19[C]
                Mol1_C_20[CH]
                Mol1_C_21[C]
                Mol1_O_7[OH]
                Mol1_C_22[CH]
                Mol1_C_23[C]
                Mol1_O_8[OH]
                Mol1_C_24[C]
        
                Mol1_O_1 === Mol1_C_1
                Mol1_C_1 --- Mol1_C_2
                Mol1_C_2 --- Mol1_O_2
                Mol1_C_2 === Mol1_C_3
                Mol1_C_3 --- Mol1_C_4
                Mol1_C_4 --- Mol1_C_5
                Mol1_C_5 --- Mol1_C_6
                Mol1_C_6 === Mol1_C_7
                Mol1_C_7 --- Mol1_O_3
                Mol1_O_3 --- Mol1_C_8
                Mol1_C_8 === Mol1_C_9
                Mol1_C_9 --- Mol1_C_10
                Mol1_C_9 --- Mol1_C_11
                Mol1_C_11 === Mol1_C_12
                Mol1_C_12 --- Mol1_C_13
                Mol1_C_13 === Mol1_C_14
                Mol1_C_5 === Mol1_C_15
                Mol1_C_15 --- Mol1_O_4
                Mol1_C_15 --- Mol1_C_16
                Mol1_C_16 === Mol1_C_17
                Mol1_C_17 --- Mol1_O_5
                Mol1_C_17 --- Mol1_C_18
                Mol1_C_3 --- Mol1_O_6
                Mol1_O_6 --- Mol1_C_19
                Mol1_C_19 === Mol1_C_20
                Mol1_C_20 --- Mol1_C_21
                Mol1_C_21 --- Mol1_O_7
                Mol1_C_21 === Mol1_C_22
                Mol1_C_22 --- Mol1_C_23
                Mol1_C_23 --- Mol1_O_8
                Mol1_C_23 === Mol1_C_24
                Mol1_C_24 --- Mol1_C_1
                Mol1_C_18 === Mol1_C_4
                Mol1_C_10 --- Mol1_C_6
                Mol1_C_14 --- Mol1_C_8
                Mol1_C_24 --- Mol1_C_19
            end
        """),

        ("mol2", """
        graph TB
            %% 原始分子名称: Mol1 (通过醚键添加呋喃环)
            subgraph Mol1["Mol1"]
                Mol1_O_1[O]
                Mol1_C_1[C]
                Mol1_C_2[C]
                Mol1_O_2[O]
                Mol1_C_3[C]
                Mol1_C_4[C]
                Mol1_C_5[C]
                Mol1_C_6[C]
                Mol1_C_7[CH]
                Mol1_O_3[O]
                Mol1_C_8[C]
                Mol1_C_9[C]
                Mol1_C_10[CH2]
                Mol1_C_11[CH]
                Mol1_C_12[CH]
                Mol1_C_13[CH]
                Mol1_C_14[CH]
                Mol1_C_15[C]
                Mol1_O_4[OH]
                Mol1_C_16[CH]
                Mol1_C_17[C]
                Mol1_O_5[OH]
                Mol1_C_18[CH]
                Mol1_O_6[O]
                Mol1_C_19[C]
                Mol1_C_20[CH]
                Mol1_C_21[C]
                Mol1_O_7[OH]
                Mol1_C_22[CH]
                Mol1_C_23[C]
                Mol1_O_8[OH]
                Mol1_C_24[C]
        
                Mol1_O_1 === Mol1_C_1
                Mol1_C_1 --- Mol1_C_2
                Mol1_C_2 --- Mol1_O_2
                Mol1_C_2 === Mol1_C_3
                Mol1_C_3 --- Mol1_C_4
                Mol1_C_4 --- Mol1_C_5
                Mol1_C_5 --- Mol1_C_6
                Mol1_C_6 === Mol1_C_7
                Mol1_C_7 --- Mol1_O_3
                Mol1_O_3 --- Mol1_C_8
                Mol1_C_8 === Mol1_C_9
                Mol1_C_9 --- Mol1_C_10
                Mol1_C_9 --- Mol1_C_11
                Mol1_C_11 === Mol1_C_12
                Mol1_C_12 --- Mol1_C_13
                Mol1_C_13 === Mol1_C_14
                Mol1_C_5 === Mol1_C_15
                Mol1_C_15 --- Mol1_O_4
                Mol1_C_15 --- Mol1_C_16
                Mol1_C_16 === Mol1_C_17
                Mol1_C_17 --- Mol1_O_5
                Mol1_C_17 --- Mol1_C_18
                Mol1_C_3 --- Mol1_O_6
                Mol1_O_6 --- Mol1_C_19
                Mol1_C_19 === Mol1_C_20
                Mol1_C_20 --- Mol1_C_21
                Mol1_C_21 --- Mol1_O_7
                Mol1_C_21 === Mol1_C_22
                Mol1_C_22 --- Mol1_C_23
                Mol1_C_23 --- Mol1_O_8
                Mol1_C_23 === Mol1_C_24
                Mol1_C_24 --- Mol1_C_1
                Mol1_C_18 === Mol1_C_4
                Mol1_C_10 --- Mol1_C_6
                Mol1_C_14 --- Mol1_C_8
                Mol1_C_24 --- Mol1_C_19
            end
        
            %% 新增呋喃环子结构
            subgraph 呋喃环["呋喃环"]
                呋喃环_C_1[CH]
                呋喃环_C_2[C]
                呋喃环_C_3[CH]
                呋喃环_C_4[CH]
                呋喃环_O_1[O]
        
                呋喃环_C_1 === 呋喃环_C_2
                呋喃环_C_2 --- 呋喃环_C_3
                呋喃环_C_3 === 呋喃环_C_4
                呋喃环_C_4 --- 呋喃环_O_1
                呋喃环_O_1 --- 呋喃环_C_1
            end
        
            %% 通过醚键连接
            Mol1_O_2 --- 呋喃环_C_2
                """),

        ("丙酮", """
        graph TB
            subgraph Acetone["Acetone"]
                Acetone_C_1[C]
                Acetone_C_2[C]
                Acetone_C_3[C]
                Acetone_O_4[O]

                Acetone_C_1 --- Acetone_C_2
                Acetone_C_2 --- Acetone_C_3
                Acetone_C_2 === Acetone_O_4
            end
        """),

        ("乙炔", """
        graph TB
            subgraph Acetylene["Acetylene"]
                Acetylene_C_1[C]
                Acetylene_C_2[C]

                Acetylene_C_1 -.- Acetylene_C_2
            end
        """),

        ("(E)-2-丁烯", """
        graph TB
            subgraph EButene["(E)-2-Butene"]
                EB_C1[C]
                EB_C2[C]
                EB_C3[C]
                EB_C4[C]

                EB_C1 --- EB_C2
                EB_C2 ===|E| EB_C3
                EB_C3 --- EB_C4
            end
        """),

        ("(Z)-2-丁烯", """
        graph TB
            subgraph ZButene["(Z)-2-Butene"]
                ZB_C1[C]
                ZB_C2[C]
                ZB_C3[C]
                ZB_C4[C]

                ZB_C1 --- ZB_C2
                ZB_C2 ===|Z| ZB_C3
                ZB_C3 --- ZB_C4
            end
        """),
    ]

    test_molecules = [
        ("mol1","""
graph TB
    %% Original molecule name: myoinositol
    subgraph myoinositol["myoinositol"]
        %% ================================
        %% Inositol-like ring (6 CH carbons)
        %% ================================
        myoinositolInositol_C_1[CH]      %% Ring carbon C1 (substituted by phosphate P1 and C6)
        myoinositolInositol_C_2_R[CH]    %% Ring carbon C2 (chiral, substituted by phosphate P2)
        myoinositolInositol_C_3_S[CH5]    %% Ring carbon C3 (stereogenic, substituted by diphosphate P3–P4)
        myoinositolInositol_C_4[CH]      %% Ring carbon C4 (bearing a single OH)
        myoinositolInositol_C_5_S[CH]    %% Ring carbon C5 (stereogenic, substituted by diphosphate P5–P6)
        myoinositolInositol_C_6_R[CH]    %% Ring carbon C6 (chiral, substituted by phosphate P7)

        %% ==========================================
        %% Phosphate at C1 (monophosphate on C1–O)
        %% P1 with one P=O and two P–OH, plus one P–O–C1 bridge
        %% ==========================================
        myoinositolPhos1_O_P1_double[O]  %% P1=O (terminal phosphoryl oxygen)
        myoinositolPhos1_P_1[P]          %% Phosphorus P1 (attached to O_P1_double, two OH, and bridge to C1)
        myoinositolPhos1_O_P1_OH1[OH]    %% P1–OH (first acidic hydroxyl)
        myoinositolPhos1_O_P1_OH2[OH]    %% P1–OH (second acidic hydroxyl)
        myoinositolPhos1_O_P1_bridge[O]  %% P1–O–C1 bridging oxygen

        %% ==========================================
        %% Phosphate at C2 (monophosphate on C2–O)
        %% P2 with one P=O and two P–OH, plus one P–O–C2 bridge
        %% ==========================================
        myoinositolPhos2_O_P1_bridge[O]  %% P2–O–C2 bridging oxygen
        myoinositolPhos2_P_1[P]          %% Phosphorus P2
        myoinositolPhos2_O_P1_double[O]  %% P2=O
        myoinositolPhos2_O_P1_OH1[OH]    %% P2–OH (first hydroxyl)
        myoinositolPhos2_O_P1_OH2[OH]    %% P2–OH (second hydroxyl)

        %% ==================================================
        %% Diphosphate at C3 (P3–O–P4 chain attached to C3–O)
        %% P3: one P=O, one P–OH, one P–O–(to C3), one P–O–P4
        %% P4: one P=O, two P–OH
        %% ==================================================
        myoinositolDiphos3_O_P1_bridge[O]       %% P3–O–C3 bridging oxygen
        myoinositolDiphos3_P_1[P]               %% Phosphorus P3 (proximal to inositol)
        myoinositolDiphos3_O_P1_double[O]       %% P3=O
        myoinositolDiphos3_O_P1_OH[OH]          %% P3–OH
        myoinositolDiphos3_O_P1_P2_bridge[O]    %% P3–O–P4 bridging oxygen
        myoinositolDiphos3_P_2[P]               %% Phosphorus P4 (distal in the diphosphate)
        myoinositolDiphos3_O_P2_double[O]       %% P4=O
        myoinositolDiphos3_O_P2_OH1[OH]         %% P4–OH (first)
        myoinositolDiphos3_O_P2_OH2[OH]         %% P4–OH (second)

        %% ==========================================
        %% C4 hydroxyl substituent (free OH, no phosphate)
        %% ==========================================
        myoinositolInositol_OH_C4[OH]           %% C4–OH

        %% ==================================================
        %% Diphosphate at C5 (P5–O–P6 chain attached to C5–O)
        %% P5: one P=O, one P–OH, one P–O–(to C5), one P–O–P6
        %% P6: one P=O, two P–OH
        %% ==================================================
        myoinositolDiphos5_O_P1_bridge[O]       %% P5–O–C5 bridging oxygen
        myoinositolDiphos5_P_1[P]               %% Phosphorus P5
        myoinositolDiphos5_O_P1_double[O]       %% P5=O
        myoinositolDiphos5_O_P1_OH[OH]          %% P5–OH
        myoinositolDiphos5_O_P1_P2_bridge[O]    %% P5–O–P6 bridging oxygen
        myoinositolDiphos5_P_2[P]               %% Phosphorus P6
        myoinositolDiphos5_O_P2_double[O]       %% P6=O
        myoinositolDiphos5_O_P2_OH1[OH]         %% P6–OH (first)
        myoinositolDiphos5_O_P2_OH2[OH]         %% P6–OH (second)

        %% ==========================================
        %% Phosphate at C6 (monophosphate on C6–O)
        %% P7 with one P=O and two P–OH, plus one P–O–C6 bridge
        %% ==========================================
        myoinositolPhos6_O_P1_bridge[O]         %% P7–O–C6 bridging oxygen
        myoinositolPhos6_P_1[P]                 %% Phosphorus P7
        myoinositolPhos6_O_P1_double[O]         %% P7=O
        myoinositolPhos6_O_P1_OH1[OH]           %% P7–OH (first)
        myoinositolPhos6_O_P1_OH2[OH]           %% P7–OH (second)

        %% ===========================
        %% Bonding (topology preserved)
        %% ===========================

        %% Phosphate at C1
        myoinositolPhos1_O_P1_double === myoinositolPhos1_P_1          %% P1=O
        myoinositolPhos1_P_1 --- myoinositolPhos1_O_P1_OH1             %% P1–OH
        myoinositolPhos1_P_1 --- myoinositolPhos1_O_P1_OH2             %% P1–OH
        myoinositolPhos1_P_1 --- myoinositolPhos1_O_P1_bridge          %% P1–O– (bridging to C1)
        myoinositolPhos1_O_P1_bridge --- myoinositolInositol_C_1       %% O–C1 (phosphomonoester at C1)

        %% Inositol-like ring: C1–C2–C3–C4–C5–C6–C1
        myoinositolInositol_C_1 --- myoinositolInositol_C_2_R          %% C1–C2
        myoinositolInositol_C_2_R --- myoinositolInositol_C_3_S        %% C2–C3
        myoinositolInositol_C_3_S --- myoinositolInositol_C_4          %% C3–C4
        myoinositolInositol_C_4 --- myoinositolInositol_C_5_S          %% C4–C5
        myoinositolInositol_C_5_S --- myoinositolInositol_C_6_R        %% C5–C6
        myoinositolInositol_C_6_R --- myoinositolInositol_C_1          %% C6–C1 (ring closure)

        %% Phosphate at C2
        myoinositolInositol_C_2_R --- myoinositolPhos2_O_P1_bridge     %% C2–O (bridge to P2)
        myoinositolPhos2_O_P1_bridge --- myoinositolPhos2_P_1          %% O–P2
        myoinositolPhos2_P_1 === myoinositolPhos2_O_P1_double          %% P2=O
        myoinositolPhos2_P_1 --- myoinositolPhos2_O_P1_OH1             %% P2–OH
        myoinositolPhos2_P_1 --- myoinositolPhos2_O_P1_OH2             %% P2–OH

        %% Diphosphate at C3 (P3–O–P4)
        myoinositolInositol_C_3_S --- myoinositolDiphos3_O_P1_bridge   %% C3–O (bridge to P3)
        myoinositolDiphos3_O_P1_bridge --- myoinositolDiphos3_P_1      %% O–P3
        myoinositolDiphos3_P_1 === myoinositolDiphos3_O_P1_double      %% P3=O
        myoinositolDiphos3_P_1 --- myoinositolDiphos3_O_P1_OH          %% P3–OH
        myoinositolDiphos3_P_1 --- myoinositolDiphos3_O_P1_P2_bridge   %% P3–O–P4 bridge
        myoinositolDiphos3_O_P1_P2_bridge --- myoinositolDiphos3_P_2   %% O–P4
        myoinositolDiphos3_P_2 === myoinositolDiphos3_O_P2_double      %% P4=O
        myoinositolDiphos3_P_2 --- myoinositolDiphos3_O_P2_OH1         %% P4–OH
        myoinositolDiphos3_P_2 --- myoinositolDiphos3_O_P2_OH2         %% P4–OH

        %% C4 hydroxyl
        myoinositolInositol_C_4 --- myoinositolInositol_OH_C4          %% C4–OH (free hydroxyl)

        %% Diphosphate at C5 (P5–O–P6)
        myoinositolInositol_C_5_S --- myoinositolDiphos5_O_P1_bridge   %% C5–O (bridge to P5)
        myoinositolDiphos5_O_P1_bridge --- myoinositolDiphos5_P_1      %% O–P5
        myoinositolDiphos5_P_1 === myoinositolDiphos5_O_P1_double      %% P5=O
        myoinositolDiphos5_P_1 --- myoinositolDiphos5_O_P1_OH          %% P5–OH
        myoinositolDiphos5_P_1 --- myoinositolDiphos5_O_P1_P2_bridge   %% P5–O–P6
        myoinositolDiphos5_O_P1_P2_bridge --- myoinositolDiphos5_P_2   %% O–P6
        myoinositolDiphos5_P_2 === myoinositolDiphos5_O_P2_double      %% P6=O
        myoinositolDiphos5_P_2 --- myoinositolDiphos5_O_P2_OH1         %% P6–OH
        myoinositolDiphos5_P_2 --- myoinositolDiphos5_O_P2_OH2         %% P6–OH

        %% Phosphate at C6
        myoinositolInositol_C_6_R --- myoinositolPhos6_O_P1_bridge     %% C6–O (bridge to P7)
        myoinositolPhos6_O_P1_bridge --- myoinositolPhos6_P_1          %% O–P7
        myoinositolPhos6_P_1 === myoinositolPhos6_O_P1_double          %% P7=O
        myoinositolPhos6_P_1 --- myoinositolPhos6_O_P1_OH1             %% P7–OH
        myoinositolPhos6_P_1 --- myoinositolPhos6_O_P1_OH2             %% P7–OH
    end



        """)]

    # 解析所有分子
    mols = []
    legends = []

    for i, (name, graph) in enumerate(test_molecules, 1):
        print(f"\n=== 测试{i}: {name} ===")
        mol,psd,mapping,warning_message = parse_mermaid_full(graph)
        print(f'case{i} warning {warning_message}')

        if mol:
            smiles = mol_to_smiles(mol)
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

            print(f"SMILES:  {smiles}")
            print(f"分子式:  {formula}")

            mols.append(mol)
            legends.append(f"{name}\n{smiles}")
        else:
            print(f"解析失败！")
            mols.append(None)
            legends.append(f"{name}\n(解析失败)")

    # 生成网格图
    print("\n" + "=" * 60)
    print("生成分子结构可视化...")
    print("=" * 60)

    # visualize_mols_grid(
    #     mols,
    #     legends=legends,
    #     mols_per_row=4,
    #     sub_img_size=(350, 350),
    #     save_path="test_molecules.png"
    # )

    print("\n所有测试完成！")
