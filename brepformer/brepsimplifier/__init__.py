"""BrepSimplifier — geometric primitive fitting for B-rep CAD models.

Decomposes complex B-rep shapes into boolean unions of canonical primitives
(box, cylinder, cone, sphere, torus) using exact OCC surface analysis.
"""

from brepformer.brepsimplifier.primitives import (
    fit_primitives,
    fit_primitives_from_step,
    defeature_and_fit,
)

__all__ = ["fit_primitives", "fit_primitives_from_step", "defeature_and_fit"]
