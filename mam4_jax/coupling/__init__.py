"""Host-model process sequencing.

E3SM couples through modal_aero_amicphys (sub-stepped, sub-area). CAM's
sequential grid-cell-mean coupling is a separate module, not yet ported --
see docs/plans/024-cesm-variant.md PR G. The two share nothing, which is why
this is a directory rather than a single module.
"""
