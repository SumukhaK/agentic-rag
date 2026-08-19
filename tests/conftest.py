import sys
from pathlib import Path

# Guarantees tests/ is importable as a plain module path (e.g.
# `from access_tiers import ...`) from every test file regardless of which
# subdirectory it lives in or whether that subdirectory has its own
# __init__.py - conftest.py is unconditionally loaded by pytest before any
# test module import, unlike relying on pytest's own implicit rootdir
# insertion (which doesn't reliably cover every subdirectory here).
sys.path.insert(0, str(Path(__file__).parent))
