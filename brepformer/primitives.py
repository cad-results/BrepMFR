#!/usr/bin/env python3
"""Re-export from brepformer.brepsimplifier.primitives.

The canonical implementation lives in brepformer/brepsimplifier/primitives.py.
This module re-exports everything so that existing imports
(e.g. from defeature_v2.py) continue to work.
"""

from brepformer.brepsimplifier.primitives import *  # noqa: F401,F403
from brepformer.brepsimplifier.primitives import main, parse_args

if __name__ == "__main__":
    main()
