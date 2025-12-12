#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Step Retrosynthesis Planning with DFS Search

This script performs multi-step retrosynthetic analysis:
1. Generate N alternative disconnection strategies for the target
2. For each strategy, perform single-step retrosynthesis
3. Recursively analyze each reactant (DFS) up to max_depth
4. Save all intermediate results to JSONL
5. Generate interactive HTML visualization of the complete synthesis tree
"""

import os
import re
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict

from rdkit import Chem
from ApiCaptioner import ApiCaptioner
from rdkit_to_mermaid import mol_to_mermaid
from mermaid_to_rdkit import mermaid_to_mol, mermaid_to_mol_with_comments
from visualize_mermaid import visualize_mermaid
from system_prompt import BASE_INSTRUCTION


# =======================
# Multi-Strategy Retrosynthesis Prompt
# =======================

MULTI_STRATEGY_PROMPT = """# Multi-Strategy Retrosynthetic Analysis

## Task Overview
You are performing **strategic retrosynthetic analysis** on a target molecule. Your goal is to propose **{num_strategies} DIFFERENT** retrosynthetic disconnection strategies, ranked by feasibility.

## Target Molecule (Mermaid Graph)

{target_mermaid}

## Critical Requirements

### 1. Multiple Strategies
- Propose **exactly {num_strategies} distinct disconnection strategies**
- Strategies should explore different bond disconnections
- Rank strategies by: synthetic feasibility, step count, reagent availability

### 2. Rich Mermaid Annotations
**EVERY atom and bond involved in the reaction MUST have inline annotations using `%%` comments**

#### Annotation Types:

**Disconnection Sites:**
```
r1_C2 --- r2_C5 %% DISCONNECTION: C-C bond, formed via aldol reaction
%% STRATEGIC BOND: Break this to form two simpler fragments
```

**Reactive Centers:**
```
r1_C2[CH2] %% NUCLEOPHILE: α-carbon, forms enolate
r2_C5[CH] %% ELECTROPHILE: carbonyl carbon, accepts attack
r1_O1[OH] %% LEAVING GROUP: departs as H2O
```

**Functional Groups:**
```
%% FG PRESENT: Ester group at C3-O2-C7
%% FG TRANSFORM: Ester → Carboxylic acid + Alcohol (hydrolysis)
```

**Stereochemistry:**
```
r1_C4[CH] %% STEREOCENTER: (R)-config, must preserve in synthesis
%% E-ALKENE: geometry controlled by elimination conditions
```

**Mechanism Hints:**
```
%% STEP 1: Base abstracts α-proton from r1_C2
%% STEP 2: Nucleophilic attack on r2_C5 carbonyl
%% STEP 3: Protonation yields β-hydroxy product
```

### 3. Complete Reactant Graphs
For each strategy, provide **fully annotated** Mermaid graphs for ALL reactants, showing:
- Which atoms are preserved from target
- Which atoms are new/modified
- All reactive sites clearly labeled
- Bond formation/breaking sites marked

## Output Format (JSON)

```json
{{
  "target_smiles": "input SMILES",
  "target_analysis": {{
    "molecular_formula": "C10H12O2",
    "key_functional_groups": ["ester", "aromatic", "methyl"],
    "complexity_score": "Medium",
    "retrosynthetic_complexity": "2-3 steps to commercial starting materials"
  }},
  
  "strategies": [
    {{
      "strategy_id": 1,
      "strategy_name": "Ester Hydrolysis + Friedel-Crafts",
      "disconnection_type": "Ester C-O bond cleavage",
      "confidence": "High",
      "estimated_yield": "75-85%",
      "reasoning": "Most straightforward route, uses common reactions...",
      
      "disconnections": [
        {{
          "bond_description": "C3-O2 ester bond",
          "bond_atoms": ["C3", "O2"],
          "reaction_type": "Ester hydrolysis",
          "annotation": "DISCONNECTION: Ester bond, reverse esterification"
        }}
      ],
      
      "reactants": [
        {{
          "reactant_id": "R1",
          "name": "Benzoic acid",
          "role": "carboxylic acid component",
          "smiles": "c1ccc(cc1)C(=O)O",
          "complexity": "Simple",
          "commercial_availability": "Yes",
          "estimated_cost": "Low",
          "mermaid_graph": "FULLY ANNOTATED Mermaid graph with %% comments"
        }},
        {{
          "reactant_id": "R2",
          "name": "Methanol",
          "role": "alcohol component",
          "smiles": "CO",
          "complexity": "Simple",
          "commercial_availability": "Yes",
          "estimated_cost": "Very Low",
          "mermaid_graph": "FULLY ANNOTATED Mermaid graph"
        }}
      ],
      
      "reagents": [
        {{
          "name": "Sulfuric acid",
          "role": "acid catalyst",
          "conditions": "catalytic amount"
        }}
      ],
      
      "conditions": {{
        "temperature": "reflux",
        "time": "2-4 hours",
        "solvent": "excess methanol",
        "atmosphere": "air"
      }},
      
      "mechanism_overview": [
        "Protonation of carbonyl oxygen",
        "Nucleophilic attack by methanol",
        "Proton transfer and loss of water",
        "Deprotonation to form ester"
      ],
      
      "pros": [
        "Simple one-step reaction",
        "High yielding",
        "Both starting materials commercially available"
      ],
      
      "cons": [
        "Requires acid catalyst",
        "Water formed must be removed for equilibrium"
      ]
    }},
    
    {{
      "strategy_id": 2,
      "strategy_name": "Alternative route 2...",
      "disconnection_type": "Different bond...",
      "confidence": "Medium",
      "reasoning": "...",
      "reactants": [...],
      "...": "..."
    }},
    
    {{
      "strategy_id": 3,
      "...": "..."
    }}
  ]
}}
```

## Quality Checklist

✓ Exactly {num_strategies} distinct strategies provided
✓ Each strategy explores different disconnections
✓ All Mermaid graphs include rich `%%` annotations
✓ Every reactive atom labeled (NUCLEOPHILE/ELECTROPHILE/LEAVING GROUP)
✓ All strategic bonds marked with DISCONNECTION annotations
✓ Functional group transformations documented
✓ Stereochemistry preserved and annotated
✓ Mechanism steps numbered and explained
✓ Commercial availability and feasibility assessed

## Your Response

Provide the complete JSON with {num_strategies} strategies and richly annotated Mermaid graphs.
"""


# =======================
# Single-Step Retrosynthesis (for recursive calls)
# =======================

SINGLE_STEP_PROMPT = """# Single-Step Retrosynthetic Analysis

## Task
Propose the **BEST SINGLE-STEP** retrosynthetic disconnection for this molecule.

## Target Molecule

{target_mermaid}

## Requirements

1. **One strategy only** - the most feasible disconnection
2. **Fully annotated Mermaid graphs** with `%%` comments for:
   - DISCONNECTION sites
   - NUCLEOPHILE/ELECTROPHILE/LEAVING GROUP labels
   - Functional group transformations
   - Stereochemistry
   - Mechanism steps

3. **Complete reactant information**:
   - Annotated Mermaid graphs
   - SMILES
   - Commercial availability
   - Complexity assessment

## Output Format (JSON)

```json
{{
  "target_smiles": "...",
  "disconnection_type": "reaction type",
  "confidence": "High/Medium/Low",
  "reasoning": "why this disconnection...",
  
  "reactants": [
    {{
      "name": "...",
      "smiles": "...",
      "role": "...",
      "complexity": "Simple/Medium/Complex",
      "commercial_availability": "Yes/No",
      "mermaid_graph": "FULLY ANNOTATED graph"
    }}
  ],
  
  "reagents": [...],
  "conditions": {{}},
  "mechanism_overview": [...]
}}
```

Provide the JSON output.
"""


# =======================
# Utility Functions
# =======================

def smiles_to_mermaid(smiles: str, name: str = "Molecule") -> Optional[str]:
    """Convert SMILES to Mermaid graph"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return mol_to_mermaid(mol, name)
    except Exception as e:
        print(f"[ERROR] SMILES to Mermaid failed: {e}")
        return None


def mermaid_to_smiles(mermaid_text: str) -> Optional[str]:
    """Convert Mermaid to SMILES"""
    try:
        mol = mermaid_to_mol(mermaid_text)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"[WARN] Mermaid to SMILES failed: {e}")
        return None


def extract_json_from_response(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response"""
    if not text:
        return None
    
    # Try JSON code block
    pattern = r"```json\s*(.+?)\s*```"
    matches = re.findall(pattern, text, flags=re.S | re.I)
    
    json_text = None
    if matches:
        json_text = matches[0].strip()
    else:
        # Try bare JSON
        pattern2 = r'\{[\s\S]*?"strategies"[\s\S]*?\][\s\S]*?\}'
        matches2 = re.findall(pattern2, text, flags=re.S)
        if not matches2:
            pattern2 = r'\{[\s\S]*?"reactants"[\s\S]*?\][\s\S]*?\}'
            matches2 = re.findall(pattern2, text, flags=re.S)
        if matches2:
            json_text = matches2[0].strip()
    
    if not json_text:
        print("[WARN] No JSON found in response")
        return None
    
    # Clean JSON
    json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        print(f"[DEBUG] Preview: {json_text[:500]}")
        return None


def call_llm(api: ApiCaptioner, prompt: str, model: str,
             temperature: float = 0.7, max_tokens: int = 46000) -> Optional[str]:
    """Call LLM with error handling"""
    try:
        response = api(
            messages=[{'text': prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        if response and "An error occurred" not in str(response):
            return response
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
    return None


def assess_molecule_complexity(smiles: str) -> str:
    """Assess if molecule is simple enough to stop recursion"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Unknown"
        
        num_atoms = mol.GetNumHeavyAtoms()
        num_rings = Chem.Descriptors.RingCount(mol)
        num_hetero = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in ['C', 'H'])
        
        # Simple molecules: few atoms, no/few rings, common building blocks
        if num_atoms <= 6 and num_rings == 0:
            return "Simple"
        elif num_atoms <= 10 and num_rings <= 1 and num_hetero <= 2:
            return "Medium"
        else:
            return "Complex"
    except:
        return "Unknown"


# =======================
# Multi-Step Retrosynthesis Core
# =======================

class RetrosynthesisNode:
    """Node in the retrosynthesis tree"""
    def __init__(self, smiles: str, mermaid: str, depth: int, parent_id: Optional[str] = None):
        self.id = f"node_{id(self)}"
        self.smiles = smiles
        self.mermaid = mermaid
        self.depth = depth
        self.parent_id = parent_id
        self.children: List['RetrosynthesisNode'] = []
        self.reactants: List[Dict] = []
        self.reagents: List[Dict] = []
        self.conditions: Dict = {}
        self.disconnection_type: str = ""
        self.confidence: str = ""
        self.is_terminal: bool = False
        self.terminal_reason: str = ""
        
    def to_dict(self) -> Dict:
        """Convert node to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "smiles": self.smiles,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "disconnection_type": self.disconnection_type,
            "confidence": self.confidence,
            "is_terminal": self.is_terminal,
            "terminal_reason": self.terminal_reason,
            "reactants": self.reactants,
            "reagents": self.reagents,
            "conditions": self.conditions,
            "num_children": len(self.children)
        }


class MultiStepRetrosynthesis:
    """Multi-step retrosynthesis planner with DFS"""
    
    def __init__(self, api: ApiCaptioner, model: str, output_dir: str,
                 max_depth: int = 3, num_strategies: int = 3):
        self.api = api
        self.model = model
        self.output_dir = Path(output_dir)
        self.max_depth = max_depth
        self.num_strategies = num_strategies
        self.all_nodes: List[RetrosynthesisNode] = []
        self.jsonl_path = self.output_dir / "retrosynthesis_steps.jsonl"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize JSONL file
        with open(self.jsonl_path, 'w', encoding='utf-8') as f:
            pass  # Clear file
    
    def save_step(self, node: RetrosynthesisNode, raw_response: str):
        """Save a retrosynthesis step to JSONL"""
        step_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "node": node.to_dict(),
            "raw_response": raw_response
        }
        
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(step_data, ensure_ascii=False) + '\n')
    
    def run_initial_analysis(self, target_smiles: str) -> List[Dict]:
        """
        Step 1: Generate multiple strategies for the target molecule
        Returns list of strategy dictionaries
        """
        print(f"\n{'='*80}")
        print(f"🎯 Initial Analysis: Generating {self.num_strategies} Strategies")
        print(f"{'='*80}\n")
        print(f"Target: {target_smiles}")
        
        # Convert to Mermaid
        target_mermaid = smiles_to_mermaid(target_smiles, "Target")
        if not target_mermaid:
            raise ValueError("Failed to convert target SMILES to Mermaid")
        
        # Build prompt
        prompt = BASE_INSTRUCTION + "\n\n" + MULTI_STRATEGY_PROMPT.format(
            num_strategies=self.num_strategies,
            target_mermaid=f"```mermaid\n{target_mermaid}\n```"
        )
        
        # Call LLM
        print(f"[INFO] Calling LLM to analyze target...")
        response = call_llm(self.api, prompt, self.model)
        if not response:
            raise RuntimeError("LLM analysis failed")
        
        # Parse strategies
        data = extract_json_from_response(response)
        if not data or "strategies" not in data:
            raise RuntimeError("Failed to parse strategies from LLM response")
        
        strategies = data["strategies"]
        print(f"✓ Generated {len(strategies)} strategies")
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n  Strategy {i}: {strategy.get('strategy_name', 'Unknown')}")
            print(f"    Confidence: {strategy.get('confidence', 'N/A')}")
            print(f"    Reactants: {len(strategy.get('reactants', []))}")
        
        return strategies
    
    def run_single_step(self, node: RetrosynthesisNode) -> bool:
        """
        Perform single-step retrosynthesis for a node
        Returns True if successful, False if terminal
        """
        print(f"\n{'─'*80}")
        print(f"🔍 Analyzing Node (Depth {node.depth})")
        print(f"{'─'*80}")
        print(f"SMILES: {node.smiles}")
        
        # Check complexity
        complexity = assess_molecule_complexity(node.smiles)
        print(f"Complexity: {complexity}")
        
        if complexity == "Simple":
            node.is_terminal = True
            node.terminal_reason = "Simple commercial building block"
            print("✓ Terminal node (simple molecule)")
            return False
        
        # Check depth limit
        if node.depth >= self.max_depth:
            node.is_terminal = True
            node.terminal_reason = f"Max depth ({self.max_depth}) reached"
            print(f"✓ Terminal node (max depth)")
            return False
        
        # Build prompt
        prompt = BASE_INSTRUCTION + "\n\n" + SINGLE_STEP_PROMPT.format(
            target_mermaid=f"```mermaid\n{node.mermaid}\n```"
        )
        
        # Call LLM
        print("[INFO] Calling LLM for single-step analysis...")
        response = call_llm(self.api, prompt, self.model)
        if not response:
            print("[WARN] LLM call failed, marking as terminal")
            node.is_terminal = True
            node.terminal_reason = "LLM analysis failed"
            return False
        
        # Parse response
        data = extract_json_from_response(response)
        if not data or "reactants" not in data:
            print("[WARN] Failed to parse response, marking as terminal")
            node.is_terminal = True
            node.terminal_reason = "Failed to parse LLM response"
            return False
        
        # Extract information
        node.disconnection_type = data.get("disconnection_type", "Unknown")
        node.confidence = data.get("confidence", "Unknown")
        node.reactants = data.get("reactants", [])
        node.reagents = data.get("reagents", [])
        node.conditions = data.get("conditions", {})
        
        print(f"✓ Disconnection: {node.disconnection_type}")
        print(f"✓ Found {len(node.reactants)} reactants")
        
        # Save step
        self.save_step(node, response)
        
        # Create child nodes for each reactant
        for reactant in node.reactants:
            reactant_smiles = reactant.get("smiles")
            reactant_mermaid = reactant.get("mermaid_graph")
            
            if not reactant_smiles or not reactant_mermaid:
                print(f"[WARN] Skipping reactant without SMILES or Mermaid")
                continue
            
            # Validate SMILES
            if not Chem.MolFromSmiles(reactant_smiles):
                print(f"[WARN] Invalid SMILES: {reactant_smiles}")
                continue
            
            child = RetrosynthesisNode(
                smiles=reactant_smiles,
                mermaid=reactant_mermaid,
                depth=node.depth + 1,
                parent_id=node.id
            )
            node.children.append(child)
            self.all_nodes.append(child)
            
            print(f"  └─ Reactant: {reactant.get('name', 'Unknown')}")
        
        return len(node.children) > 0
    
    def dfs_explore(self, node: RetrosynthesisNode):
        """Depth-first search exploration of retrosynthesis tree"""
        # Process current node
        has_children = self.run_single_step(node)
        
        if not has_children:
            return
        
        # Recursively explore children
        for child in node.children:
            self.dfs_explore(child)
    
    def run(self, target_smiles: str) -> Dict[str, Any]:
        """
        Run complete multi-step retrosynthesis
        
        Returns:
            Dictionary with analysis results and paths
        """
        start_time = time.time()
        
        # Step 1: Generate initial strategies
        strategies = self.run_initial_analysis(target_smiles)
        
        # Step 2: For each strategy, build a retrosynthesis tree
        trees = []
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n{'='*80}")
            print(f"🌲 Exploring Strategy {i}/{len(strategies)}: {strategy.get('strategy_name', 'Unknown')}")
            print(f"{'='*80}")
            
            # Create root node for this strategy
            root = RetrosynthesisNode(
                smiles=target_smiles,
                mermaid=smiles_to_mermaid(target_smiles, f"Target_S{i}"),
                depth=0
            )
            root.disconnection_type = strategy.get("disconnection_type", "")
            root.confidence = strategy.get("confidence", "")
            
            # Initialize reactants from strategy
            for reactant_data in strategy.get("reactants", []):
                reactant_smiles = reactant_data.get("smiles")
                reactant_mermaid = reactant_data.get("mermaid_graph")
                
                if not reactant_smiles or not reactant_mermaid:
                    continue
                
                child = RetrosynthesisNode(
                    smiles=reactant_smiles,
                    mermaid=reactant_mermaid,
                    depth=1,
                    parent_id=root.id
                )
                root.children.append(child)
                self.all_nodes.append(child)
            
            # Save initial strategy
            self.save_step(root, json.dumps(strategy, ensure_ascii=False))
            
            # DFS exploration from each reactant
            for child in root.children:
                self.dfs_explore(child)
            
            trees.append({
                "strategy_id": i,
                "strategy_name": strategy.get("strategy_name"),
                "root": root,
                "total_nodes": len([n for n in self.all_nodes if n.depth > 0])
            })
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"✅ Multi-Step Retrosynthesis Complete!")
        print(f"{'='*80}")
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"🌲 Strategies explored: {len(trees)}")
        print(f"📊 Total nodes: {len(self.all_nodes)}")
        print(f"💾 Steps saved to: {self.jsonl_path}")
        
        return {
            "target_smiles": target_smiles,
            "num_strategies": len(strategies),
            "trees": trees,
            "total_nodes": len(self.all_nodes),
            "jsonl_path": str(self.jsonl_path),
            "elapsed_time": elapsed
        }


# =======================
# Visualization Generation
# =======================

def _build_tooltips(meta: Dict) -> Dict[str, Dict[int, str]]:
    """Build tooltips from metadata"""
    atom_idx_map = meta.get("atom_id_to_idx", {}) or {}
    bond_idx_map = meta.get("bond_key_to_idx", {}) or {}
    atom_comments = meta.get("atoms", {}) or {}
    bond_comments = meta.get("bonds", {}) or {}

    atoms = {
        atom_idx_map[atom_id]: comment
        for atom_id, comment in atom_comments.items()
        if comment and atom_id in atom_idx_map
    }
    bonds = {
        bond_idx_map[bond_key]: comment
        for bond_key, comment in bond_comments.items()
        if comment and bond_key in bond_idx_map
    }
    return {"atoms": atoms, "bonds": bonds}


def _to_js_safe_json(data: Dict) -> str:
    """JSON safe for embedding in HTML"""
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def generate_molecule_viewer_data(mermaid_text: str, molecule_id: str,
                                   size: Tuple[int, int] = (600, 450)) -> Dict:
    """Generate SVG variants and tooltip data for a molecule"""
    try:
        mol, meta = mermaid_to_mol_with_comments(mermaid_text, strict=False, respect_h_count=True)
        if mol is None:
            return {"error": "Failed to parse Mermaid"}

        tooltip_data = _build_tooltips(meta)
        highlight_atoms = sorted(tooltip_data.get("atoms", {}).keys())
        highlight_bonds = sorted(tooltip_data.get("bonds", {}).keys())

        def _svg(show_atoms: bool, show_bonds: bool) -> str:
            svg = visualize_mermaid(
                mermaid_text,
                size=size,
                strict=False,
                respect_h_count=True,
                show_atom_notes=show_atoms,
                show_bond_notes=show_bonds,
                highlight_atoms=highlight_atoms,
                highlight_bonds=highlight_bonds,
                adjust_atom_labels=False,
            )
            return svg if svg else ""

        svg_variants = {
            "0|0": _svg(False, False),
            "1|0": _svg(True, False),
            "0|1": _svg(False, True),
            "1|1": _svg(True, True),
        }

        return {
            "id": molecule_id,
            "svg_variants": svg_variants,
            "tooltip_data": tooltip_data,
            "has_annotations": bool(highlight_atoms or highlight_bonds)
        }
    except Exception as e:
        print(f"[WARN] Failed to generate viewer for {molecule_id}: {e}")
        return {"error": str(e), "id": molecule_id}


def generate_tree_visualization(result: Dict, output_path: str):
    """Generate interactive HTML visualization of all retrosynthesis trees"""
    
    target_smiles = result["target_smiles"]
    trees = result["trees"]
    
    # Load all steps from JSONL
    steps_by_node = {}
    with open(result["jsonl_path"], 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                step = json.loads(line)
                node_id = step["node"]["id"]
                steps_by_node[node_id] = step
    
    # Build molecule viewer data for all unique molecules
    print("\n[INFO] Generating interactive molecule viewers...")
    molecule_viewers = {}
    processed_smiles = set()
    
    for tree in trees:
        root = tree["root"]
        nodes_to_process = [root] + root.children
        
        while nodes_to_process:
            node = nodes_to_process.pop(0)
            
            if node.smiles not in processed_smiles:
                viewer_data = generate_molecule_viewer_data(
                    node.mermaid,
                    f"mol_{len(molecule_viewers)}",
                    size=(600, 450)
                )
                if "error" not in viewer_data:
                    molecule_viewers[node.smiles] = viewer_data
                    processed_smiles.add(node.smiles)
            
            nodes_to_process.extend(node.children)
    
    print(f"✓ Generated {len(molecule_viewers)} molecule viewers")
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Step Retrosynthesis Tree</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .header .smiles {{
            font-family: 'Courier New', monospace;
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 8px;
            display: inline-block;
            margin-top: 15px;
        }}
        
        .stats {{
            display: flex;
            justify-content: space-around;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .strategy-section {{
            margin-bottom: 50px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 12px;
            border-left: 5px solid #667eea;
        }}
        
        .strategy-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }}
        
        .strategy-header h2 {{
            color: #667eea;
            font-size: 1.8em;
        }}
        
        .badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        
        .badge.high {{ background: #d4edda; color: #155724; }}
        .badge.medium {{ background: #fff3cd; color: #856404; }}
        .badge.low {{ background: #f8d7da; color: #721c24; }}
        
        .tree-container {{
            margin-top: 30px;
        }}
        
        .tree-node {{
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .tree-node.depth-0 {{ border-left: 5px solid #667eea; }}
        .tree-node.depth-1 {{ border-left: 5px solid #28a745; margin-left: 40px; }}
        .tree-node.depth-2 {{ border-left: 5px solid #ffc107; margin-left: 80px; }}
        .tree-node.depth-3 {{ border-left: 5px solid #dc3545; margin-left: 120px; }}
        
        .node-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .node-title {{
            font-weight: bold;
            color: #495057;
        }}
        
        .node-depth {{
            background: #6c757d;
            color: white;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.85em;
        }}
        
        .molecule-viewer {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }}
        
        .viewer-controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        
        .viewer-controls button {{
            padding: 8px 16px;
            border: 1px solid #dee2e6;
            background: white;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .viewer-controls button:hover {{
            background: #e9ecef;
        }}
        
        .viewer-controls button.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}
        
        .svg-container {{
            background: white;
            border-radius: 6px;
            padding: 15px;
            display: flex;
            justify-content: center;
            overflow-x: auto;
        }}
        
        .reaction-info {{
            margin-top: 15px;
            padding: 15px;
            background: #e7f3ff;
            border-radius: 6px;
        }}
        
        .reaction-info h4 {{
            color: #0056b3;
            margin-bottom: 10px;
        }}
        
        .reaction-info p {{
            margin: 5px 0;
            color: #495057;
        }}
        
        .terminal-node {{
            background: #d4edda;
            border: 2px solid #28a745;
        }}
        
        .terminal-badge {{
            background: #28a745;
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            margin-left: 10px;
        }}
        
        .tooltip {{
            position: fixed;
            pointer-events: none;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 9999;
            max-width: 300px;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌲 Multi-Step Retrosynthesis Tree</h1>
            <div class="smiles">{target_smiles}</div>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value">{result['num_strategies']}</div>
                <div class="stat-label">Strategies Explored</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{result['total_nodes']}</div>
                <div class="stat-label">Total Nodes</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{result.get('elapsed_time', 0):.1f}s</div>
                <div class="stat-label">Analysis Time</div>
            </div>
        </div>
        
        <div class="content">
"""
    
    # Add each strategy tree
    for i, tree in enumerate(trees, 1):
        root = tree["root"]
        strategy_name = tree.get("strategy_name", f"Strategy {i}")
        
        html_content += f"""
            <div class="strategy-section">
                <div class="strategy-header">
                    <h2>Strategy {i}: {strategy_name}</h2>
                    <span class="badge {root.confidence.lower()}">{root.confidence}</span>
                </div>
                
                <div class="tree-container">
"""
        
        # DFS traversal to render nodes
        def render_node(node, html):
            node_class = "tree-node"
            if node.is_terminal:
                node_class += " terminal-node"
            node_class += f" depth-{node.depth}"
            
            html += f"""
                    <div class="{node_class}">
                        <div class="node-header">
                            <span class="node-title">
                                {node.smiles}
                                {"<span class='terminal-badge'>✓ Terminal</span>" if node.is_terminal else ""}
                            </span>
                            <span class="node-depth">Depth {node.depth}</span>
                        </div>
"""
            
            # Add molecule viewer
            if node.smiles in molecule_viewers:
                viewer = molecule_viewers[node.smiles]
                viewer_id = f"viewer_s{i}_d{node.depth}_{id(node)}"
                
                html += f"""
                        <div class="molecule-viewer">
                            <div class="viewer-controls">
                                <button onclick="toggleAnnotations('{viewer_id}', 'atoms')">
                                    Toggle Atom Annotations
                                </button>
                                <button onclick="toggleAnnotations('{viewer_id}', 'bonds')">
                                    Toggle Bond Annotations
                                </button>
                            </div>
                            <div class="svg-container" id="{viewer_id}">
                                {viewer['svg_variants']['0|0']}
                            </div>
                        </div>
"""
            
            # Add reaction info
            if node.disconnection_type:
                html += f"""
                        <div class="reaction-info">
                            <h4>⚗️ Disconnection: {node.disconnection_type}</h4>
                            <p><strong>Confidence:</strong> {node.confidence}</p>
"""
                if node.reactants:
                    html += f"<p><strong>Products:</strong> {len(node.reactants)} reactants</p>"
                if node.terminal_reason:
                    html += f"<p><strong>Terminal Reason:</strong> {node.terminal_reason}</p>"
                html += "</div>"
            
            html += "</div>\n"
            
            # Recursively render children
            for child in node.children:
                html = render_node(child, html)
            
            return html
        
        html_content = render_node(root, html_content)
        
        html_content += """
                </div>
            </div>
"""
    
    # Add JavaScript for interactivity
    html_content += """
        </div>
    </div>
    
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        // Store all molecule viewer data
        const moleculeViewers = """ + _to_js_safe_json(molecule_viewers) + """;
        
        // Track annotation state for each viewer
        const viewerStates = {};
        
        function toggleAnnotations(viewerId, type) {
            if (!viewerStates[viewerId]) {
                viewerStates[viewerId] = { atoms: false, bonds: false };
            }
            
            viewerStates[viewerId][type] = !viewerStates[viewerId][type];
            updateViewer(viewerId);
        }
        
        function updateViewer(viewerId) {
            const container = document.getElementById(viewerId);
            if (!container) return;
            
            const state = viewerStates[viewerId] || { atoms: false, bonds: false };
            const key = `${state.atoms ? 1 : 0}|${state.bonds ? 1 : 0}`;
            
            // Find the molecule data
            for (const smiles in moleculeViewers) {
                const viewer = moleculeViewers[smiles];
                if (container.innerHTML.includes(viewer.id)) {
                    container.innerHTML = viewer.svg_variants[key];
                    break;
                }
            }
        }
        
        // Tooltip functionality
        const tooltip = document.getElementById('tooltip');
        
        document.addEventListener('mouseover', (e) => {
            const target = e.target;
            const title = target.getAttribute('title') || target.querySelector('title')?.textContent;
            
            if (title) {
                tooltip.textContent = title;
                tooltip.style.opacity = '1';
                updateTooltipPosition(e);
            }
        });
        
        document.addEventListener('mousemove', updateTooltipPosition);
        
        document.addEventListener('mouseout', (e) => {
            tooltip.style.opacity = '0';
        });
        
        function updateTooltipPosition(e) {
            tooltip.style.left = `${e.clientX + 15}px`;
            tooltip.style.top = `${e.clientY + 15}px`;
        }
    </script>
</body>
</html>
"""
    
    # Write HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✓ Interactive visualization saved to: {output_path}")


# =======================
# Main Entry Point
# =======================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Step Retrosynthesis Planning with DFS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze aspirin with 3 strategies and max depth 3
  python multi_step_retro.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
  
  # Custom parameters
  python multi_step_retro.py --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \\
      --strategies 2 --max_depth 2 --output_dir my_retro
        """
    )
    
    parser.add_argument("--smiles", type=str, required=True,
                       help="Target molecule SMILES")
    parser.add_argument("--strategies", type=int, default=3,
                       help="Number of initial strategies to explore")
    parser.add_argument("--max_depth", type=int, default=3,
                       help="Maximum recursion depth")
    parser.add_argument("--output_dir", type=str, default="multi_step_results",
                       help="Output directory")
    
    # API config
    parser.add_argument("--api_key", type=str,
                       default="sk-zPH1VHswT0F19RgzUAGFtX6UmJed8M6qo4Ts5cj7aJQNUORp")
    parser.add_argument("--base_url", type=str,
                       default="https://api.openai-proxy.org/v1")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview")
    
    args = parser.parse_args()
    
    # Initialize API
    api = ApiCaptioner(key=args.api_key, base_url=args.base_url)
    
    # Run multi-step retrosynthesis
    planner = MultiStepRetrosynthesis(
        api=api,
        model=args.model,
        output_dir=args.output_dir,
        max_depth=args.max_depth,
        num_strategies=args.strategies
    )
    
    try:
        result = planner.run(args.smiles)
        
        # Generate visualization
        html_path = os.path.join(args.output_dir, "retrosynthesis_tree.html")
        generate_tree_visualization(result, html_path)
        
        print(f"\n🌐 Open visualization in browser:")
        print(f"   file://{os.path.abspath(html_path)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()
