import numpy as np
import pytest

from vstent.imaging.qca_tree import tree_qca
from vstent.imaging.synthetic import render_tubes, stenosis_diameter
from vstent.inference.pipeline import analyse_tree
from vstent.physics.solver import PhysicsParams
from vstent.physics.tree import solve_tree
from vstent.planning.stent import Stent, apply_path_stents
from vstent.units import mm_to_m

PARAMS = PhysicsParams()
MM_PER_PX = 0.25


def _line(p0, p1, n=2000):
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - t) * np.asarray(p0, float) + t * np.asarray(p1, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return pts, s


def y_vessel():
    """Main vessel (12 px, 60% narrowing at col 300) with a 45° side branch (8 px) taking off
    at col 150. Returns the probability-like image."""
    main_pts, main_s = _line((100, 10), (100, 390))
    main_d = stenosis_diameter(main_s, 12.0, 0.6, center_px=290.0, length_px=30.0)
    side_pts, side_s = _line((100, 150), (100 + 140, 150 + 140))
    side_d = np.full_like(side_s, 8.0)
    img = render_tubes((260, 400), [(main_pts, main_d), (side_pts, side_d)], seed=1)
    return img


def test_tree_qca_finds_the_side_branch_and_its_take_off():
    tq = tree_qca(y_vessel(), (100, 12), (100, 388), mm_per_px=MM_PER_PX)
    tree = tq.tree
    assert tree.n_branches == 2 and tree.parent == (-1, 0)
    main, side = tq.branches
    # the take-off sample on the main centreline is near column 150. A skeleton's junction
    # sits up to about one vessel width downstream of where the centrelines cross (here 158).
    take_off = main.centerline.points[tree.attach[1]]
    assert abs(take_off[1] - 150) < 12 and abs(take_off[0] - 100) < 2
    # the side branch is measured from the main's wall: ~8 px wide, ~185 px long minus the wall
    assert np.median(side.diameters.diameter_px) == pytest.approx(8.0, abs=1.0)
    assert 150 < side.centerline.length < 200
    assert side.centerline.points[0][0] > 100 + 5  # starts outside the main lumen
    # the main path still sees its narrowing
    assert main.max_percent_stenosis > 45


def test_analyse_tree_side_branch_raises_main_ffr():
    res = analyse_tree(y_vessel(), (100, 12), (100, 388), params=PARAMS, mm_per_px=MM_PER_PX)
    tree_ffr, single_ffr = res.result.ffr[0][-1], res.single.ffr[-1]
    assert tree_ffr > single_ffr + 0.01  # the side branch takes flow before the narrowing
    assert res.result.q_outlet[1] > 0
    assert not any("Pixel size" in w for w in res.warnings)


def test_stents_on_a_path():
    res = analyse_tree(y_vessel(), (100, 12), (100, 388), params=PARAMS, mm_per_px=MM_PER_PX)
    tree, pre = res.qca.tree, res.result
    main = tree.branches[0]
    # stent over the main narrowing (path of the main outlet)
    throat = float(main.s[np.argmin(main.r)])
    stent = Stent(throat - mm_to_m(6), mm_to_m(12), 2 * float(main.r_ref[np.argmin(main.r)]))
    post = solve_tree(apply_path_stents(tree, 0, [stent]), PARAMS, r_outlet=pre.r_outlet)
    assert post.ffr[0][-1] > pre.ffr[0][-1] + 0.05
    # a stent on the side branch's path, entirely past the take-off: the main is unchanged
    offset = tree.path_offsets(1)[-1][3]
    side_stent = Stent(offset + mm_to_m(5), mm_to_m(8), 2.0e-3)
    stented = apply_path_stents(tree, 1, [side_stent])
    np.testing.assert_array_equal(stented.branches[0].r, tree.branches[0].r)
    assert np.any(stented.branches[1].r > tree.branches[1].r)
    with pytest.raises(ValueError, match="outside the path"):
        apply_path_stents(tree, 1, [Stent(tree.path_length(1), mm_to_m(8), 2e-3)])


def test_path_offsets():
    tree = tree_qca(y_vessel(), (100, 12), (100, 388), mm_per_px=MM_PER_PX).tree
    pieces = tree.path_offsets(1)
    assert [p[0] for p in pieces] == [0, 1]
    assert pieces[1][3] == pytest.approx(float(tree.branches[0].s[tree.attach[1]]))
    assert tree.path_length(1) == pytest.approx(pieces[1][3] + tree.branches[1].length)


def test_cli_analyse_tree_with_a_stent(dark_seg_dir, tmp_path):
    import cv2
    from typer.testing import CliRunner

    from vstent.cli import app

    path = tmp_path / "y.png"
    cv2.imwrite(str(path), ((1.0 - 0.6 * y_vessel()) * 255).astype(np.uint8))
    base = ["analyse", str(path), "--seg-model", str(dark_seg_dir), "--tree",
            "--start", "100", "12", "--end", "100", "388", "--mm-per-px", "0.25"]  # fmt: skip
    # the narrowing is ~67 mm along the main path; stent 61-73 mm, 3.0 mm
    args = base + ["--stent", "61", "12", "3.0", "--figure", str(tmp_path / "tree.png")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Vessel tree: 2 branches" in out and "Main-path distal FFR: tree" in out
    assert "Virtual stent 3.00 × 12 mm" in out and (tmp_path / "tree.png").stat().st_size > 0
    assert CliRunner().invoke(app, base + ["--plan"]).exit_code == 2
    no_tree = [a for a in base if a != "--tree"] + ["--stent", "61", "12", "3.0"]
    assert CliRunner().invoke(app, no_tree).exit_code == 2
