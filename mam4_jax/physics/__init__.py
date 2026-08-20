"""The microphysics, one module per process.

Each is a port of a Fortran subroutine; see the module docstrings for the
port target. Kernels (coag, newnuc, kohler, saturation) and the two
process-level drivers (calcsize, wateruptake) live together because the
previous split between them was arbitrary.
"""
