import os
import sys

# Extend src.__path__ to include Day 2 and Day 3 src modules seamlessly
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_WEEK3_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))

_D2_SRC = os.path.join(_WEEK3_DIR, 'day 2', 'src')
_D3_SRC = os.path.join(_WEEK3_DIR, 'day 3', 'src')

for p in [_D2_SRC, _D3_SRC]:
    if os.path.exists(p) and p not in __path__:
        __path__.append(p)

from src.state import AFLGraphState, create_initial_state
from src.graph import build_afl_graph, get_afl_graph, run_afl_turn

__all__ = [
    "AFLGraphState",
    "create_initial_state",
    "build_afl_graph",
    "get_afl_graph",
    "run_afl_turn"
]
