import numpy as np
import pytest

from vstent.units import PA_PER_MMHG, m_to_mm, mm_to_m, mmhg_to_pa, pa_to_mmhg


def test_unit_conversions():
    assert PA_PER_MMHG == 133.322
    assert mmhg_to_pa(100.0) == pytest.approx(13332.2)
    assert mm_to_m(3.0) == pytest.approx(0.003)
    x = np.array([0.5, 3.0, 93.0])
    assert np.allclose(pa_to_mmhg(mmhg_to_pa(x)), x) and np.allclose(m_to_mm(mm_to_m(x)), x)
