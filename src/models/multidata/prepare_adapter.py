"""Make a traceable, numerical-only adaptation of the downloaded author code.

No optimisation method bodies are rewritten. Unsupported categorical and
comparison-method imports are excluded; this is not a full repository repair.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import os
SOURCE = Path(os.environ['DCFO_AUTHOR_DIR'])
ROOT = Path(os.environ['DCFO_ADAPTER_DIR']); ROOT.mkdir(parents=True,exist_ok=True)
DEST = ROOT / 'numeric_modules'
DEST.mkdir(exist_ok=True)
(DEST / '__init__.py').write_text('', encoding='utf-8')

def extract(filename, names):
    source = (SOURCE / 'modules' / filename).read_text(encoding='utf-8')
    tree = ast.parse(source)
    parts = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            parts.append(ast.get_source_segment(source, node))
    assert len(parts) == len(names)
    return '\n\n'.join(parts)

common = '''from __future__ import annotations
import numpy as np
import pandas as pd
import time
from typing import Optional, Iterable, Union, List
from sklearn.neighbors import NearestNeighbors
'''
utility_names = ['preprocessOptimization', 'euclidean_distance', 'constraintLof']
(DEST / 'utilityFunctions.py').write_text(common + '\n' + extract('utilityFunctions.py', utility_names) + "\nmetricToDistance = {'minkowski': euclidean_distance, 'euclidean': euclidean_distance}\n", encoding='utf-8')
(DEST / 'lof.py').write_text(common + '\n' + extract('lof.py', ['customLOF']).replace('    @typechecked\n', ''), encoding='utf-8')
optimizer = common + '''
from scipy.optimize import minimize
from scipy.special import erf
from collections import deque, OrderedDict
from copy import copy
from .utilityFunctions import preprocessOptimization, euclidean_distance, constraintLof
'''
optimizer += '\n' + extract('optimizationLofProper.py', ['OptimizationLOF'])
(DEST / 'optimizationLofProper.py').write_text(optimizer, encoding='utf-8')
manifest = {name: hashlib.sha256((SOURCE / 'modules' / name).read_bytes()).hexdigest() for name in ['lof.py', 'utilityFunctions.py', 'optimizationLofProper.py']}
(ROOT / 'source_manifest.json').write_text(json.dumps({'source_folder': str(SOURCE), 'repository': 'https://github.com/tommasoamico/DCFO', 'sha256': manifest, 'scope': 'Numeric Euclidean demo only; extracted class/function bodies unchanged; alternative dependencies excluded.'}, indent=2), encoding='utf-8')
print('Prepared numerical-only adapter:', DEST)
