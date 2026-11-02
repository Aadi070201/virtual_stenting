"""P4h: the densitometric lumen reads tight lesions that edges cannot resolve."""

import numpy as np
import pytest

from vstent.imaging import lumen_report as lr
from vstent.imaging.centerline import smooth_and_resample
from vstent.imaging.densitometry import integrated_contrast
from vstent.imaging.qca import qca_from_centerline, robust_reference


def _ds(case, lumen):
    cl = smooth_and_resample(np.round(case["points"][::8]))
    q = qca_from_centerline(case["mask"], cl, image=case["image"], lumen=lumen,
                            mm_per_px=case["imaging"]["mm_per_px"])  # fmt: skip
    return q.max_percent_stenosis, q


def test_dense_lumen_resolves_a_tight_lesion():
    case = lr.render_lesion(8.0, 0.85, "circular", 6100)  # seeds of neither P4h render set
    edges, _ = _ds(case, "edges")
    dense, q = _ds(case, "dense_log")
    assert edges < 60.0 and abs(dense - 85.0) <= 12.0
    assert q.lumen_method == "dense"
    healthy, _ = _ds(lr.render_lesion(12.0, 0.0, "none", 6101), "dense_log")
    assert healthy < 15.0


def test_integrated_contrast_scales_with_area():
    """Without noise or edge enhancement, the IOD ratio is the area ratio (blur keeps it)."""
    from vstent.imaging.synthetic import render_projection, sine_centerline, stenosis_diameter

    pts, s = sine_centerline((100.0, 20.0), lr.LENGTH_PX, amplitude_px=10.0, wavelength_px=300.0)
    d = stenosis_diameter(s, 12.0, 0.5, s[-1] / 2, 40.0)
    img = render_projection(lr.SHAPE, [(pts, d / 2, d / 2)], mm_per_px=0.26, mu_per_mm=0.06,
                            background=0.56, blur_sigma=1.0, edge_amount=0.0, edge_sigma=2.0,
                            noise_std=0.0, seed=1)  # fmt: skip
    img8 = np.round(img * 255).astype(np.uint8)
    cl = smooth_and_resample(np.round(pts[::8]))
    q = qca_from_centerline(np.ones(lr.SHAPE), cl, image=img8, lumen="edges")
    ref = robust_reference(q.centerline.s, q.diameters.diameter_px)
    iod = integrated_contrast(img8, q.centerline, ref, "log")
    assert iod.min() / np.median(iod) == pytest.approx(0.25, abs=0.06)  # (1 − 0.5)²
    with pytest.raises(ValueError, match="domain"):
        integrated_contrast(img8, q.centerline, ref, "gamma")
