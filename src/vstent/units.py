"""Unit conversions. Everything inside vstent is SI; convert only at input/output boundaries."""

PA_PER_MMHG = 133.322
M_PER_MM = 1e-3


def mmhg_to_pa(p):
    """Pressure: mmHg -> Pa (scalar or array)."""
    return p * PA_PER_MMHG


def pa_to_mmhg(p):
    """Pressure: Pa -> mmHg (scalar or array)."""
    return p / PA_PER_MMHG


def mm_to_m(x):
    """Length: mm -> m (scalar or array)."""
    return x * M_PER_MM


def m_to_mm(x):
    """Length: m -> mm (scalar or array)."""
    return x / M_PER_MM
