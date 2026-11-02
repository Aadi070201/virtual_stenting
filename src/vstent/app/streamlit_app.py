"""vstent Streamlit app (``vstent app``). Research use only — not for clinical decisions.

Synthetic mode: a preset or random synthetic vessel. Image mode: upload a PNG / JPG / DICOM
(or pick an ARCADE example), see the segmentation, click start and end points, set the
calibration. Then: diameter and FFR plots (solver and surrogate), a virtual stent with instant
post-PCI FFR, the plan optimiser, and a case report export. Distal FFR comes with 90% intervals
(surrogate ensemble: conformal; inputs: Monte Carlo through the solver) and grey-zone flags.
Slow steps (segmentation, QCA, the pre-PCI Monte Carlo) are cached so slider updates only
recompute the stent, its post-PCI draws and the plots.
"""

import hashlib
import time

_T0 = time.perf_counter()

import numpy as np  # noqa: E402

# Plotly and st.dataframe import pandas / pyarrow lazily, the first time a chart or table is
# drawn. If the user clicks while that first import is running, Streamlit stops the script
# mid-import and leaves a half-initialised module behind ("partially initialized module
# 'pandas'"), breaking every later chart. Importing them here, before any widget exists,
# closes that window.
import pandas  # noqa: E402, F401
import pyarrow  # noqa: E402, F401
import streamlit as st  # noqa: E402

from vstent.app import core, tree_view  # noqa: E402
from vstent.imaging.centerline import longest_path_endpoints  # noqa: E402
from vstent.imaging.flow import combine_runs  # noqa: E402
from vstent.imaging.image_io import (  # noqa: E402
    ImageInput,
    is_multiframe_dicom,
    read_image,
    read_run_bytes,
)
from vstent.imaging.run_analysis import analyse_frames, pixel_distance  # noqa: E402
from vstent.imaging.segment import THRESHOLD  # noqa: E402
from vstent.inference.pipeline import (  # noqa: E402
    AUTO_LUMEN,
    analyse_tree,
    analyse_two_views,  # noqa: E402
    analyse_vessel,
    range_warnings,
)
from vstent.inference.tree_runtime import (  # noqa: E402
    TreeSurrogateRunner,
    floor_warnings,
)
from vstent.physics.tree import solve_tree  # noqa: E402
from vstent.planning.optimizer import plan_stents  # noqa: E402
from vstent.planning.stent import Stent, apply_path_stents  # noqa: E402
from vstent.planning.tree_planner import plan_tree_stents  # noqa: E402
from vstent.uncertainty.greyzone import load_uncertainty, plan_assessments  # noqa: E402
from vstent.uncertainty.inputs import SOURCE_LABELS, Measurement  # noqa: E402
from vstent.units import m_to_mm, mm_to_m, mmhg_to_pa, pa_to_mmhg  # noqa: E402

DISPLAY_WIDTH = 512
LUMEN_LABELS = {
    "scaled": "Intensity edges, size from the imaging chain",
    "edges": "Intensity edges",
    "model": "Imaging-chain model",
}
LUMEN_CHOICES = {LUMEN_LABELS[k]: k for k in sorted(LUMEN_LABELS, key=lambda k: k != AUTO_LUMEN)}
DICOM_CALIBRATION = "DICOM geometry (at the isocentre)"
TWO_VIEW_SOURCE = "Two-view example (synthetic)"

st.set_page_config(page_title="vstent research prototype", layout="wide")
st.error(core.BANNER, icon="⚠️")


@st.cache_resource
def get_models() -> core.Models:
    return core.load_models()


@st.cache_resource
def get_settings():
    return core.load_settings()


@st.cache_resource(max_entries=8, show_spinner="Segmenting the angiogram…")
def segment(sha: str, _image: np.ndarray) -> np.ndarray:
    return get_models().segmentation.predict(_image)


@st.cache_resource(max_entries=4, show_spinner="Segmenting every frame of the run…")
def run_analysis(sha: str, _run):
    return analyse_frames(_run, get_models().segmentation.predict)


@st.cache_resource(max_entries=16, show_spinner="Measuring the flow velocity…")
def run_velocity(key: str, _rf, _centerline, frame: int, mm_per_px, hyperemic: bool,
                 _view_s=None, _s=None):  # fmt: skip
    """One run's flow velocity along the centreline: in one view (``mm_per_px``; ``None``: not
    calibrated) or along the 3D path (``_view_s`` → ``_s``)."""
    if _view_s is None:
        return _rf.velocity(_centerline, frame, pixel_distance(mm_per_px), single_view=True,
                            hyperemic_run=hyperemic)  # fmt: skip
    return _rf.velocity(_centerline, frame, lambda s: np.interp(s, _view_s, _s),
                        single_view=False, hyperemic_run=hyperemic)  # fmt: skip


@st.cache_resource(max_entries=32, show_spinner="Measuring the vessel (QCA)…")
def analyse(sha: str, start, end, mm_per_px, pa, pv, lumen, _prob, _image, v_hyp=None):
    # the user's pressures, so the solver, R_micro and the range warnings all use them
    pressures = {"pa_mmhg": pa_to_mmhg(pa), "pv_mmhg": pa_to_mmhg(pv)}
    params = get_settings()[0].model_copy(update=pressures | ({"v_hyp": v_hyp} if v_hyp else {}))
    return analyse_vessel(
        _prob, start, end, params=params, surrogate=get_models().surrogate, mm_per_px=mm_per_px,
        image=_image, lumen=lumen,
    )  # fmt: skip


@st.cache_resource(show_spinner="Rendering the synthetic two-view example…")
def get_two_view_example() -> core.TwoViewExample:
    return core.two_view_example()


@st.cache_resource(max_entries=16, show_spinner="Reconstructing the vessel in 3D (two views)…")
def analyse_two(key: str, starts, ends, pa, pv, lumen, _probs, _images, _geometries,
                v_hyp=None):  # fmt: skip
    pressures = {"pa_mmhg": pa_to_mmhg(pa), "pv_mmhg": pa_to_mmhg(pv)}
    params = get_settings()[0].model_copy(update=pressures | ({"v_hyp": v_hyp} if v_hyp else {}))
    return analyse_two_views(
        _probs, starts, ends, _geometries, params=params, surrogate=get_models().surrogate,
        images=_images, lumen=lumen,
    )  # fmt: skip


def _click_points(overlay, shape, key: str, names: tuple[str, str], last: str) -> None:
    """Show ``overlay``; a click sets the first point, then the second (``names`` in the
    session state), and starts again after both."""
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates

        click = streamlit_image_coordinates(overlay, width=DISPLAY_WIDTH, key=key)
    except Exception:  # component unavailable (e.g. in tests): static image
        st.image(overlay, width=DISPLAY_WIDTH)
        click = None
    if click and click != st.session_state.get(last):
        st.session_state[last] = click
        h, w = shape
        pt = (int(round(click["y"] * h / click["height"])),
              int(round(click["x"] * w / click["width"])))  # fmt: skip
        first, second = names
        if st.session_state.get(first) is None or st.session_state.get(second) is not None:
            st.session_state.update({first: pt, second: None})
        else:
            st.session_state[second] = pt
        st.rerun()


@st.cache_resource(max_entries=16, show_spinner="Measuring the vessel tree…")
def analyse_tree_cached(sha: str, start, end, mm_per_px, pa, pv, lumen, _prob, _image):
    pressures = {"pa_mmhg": pa_to_mmhg(pa), "pv_mmhg": pa_to_mmhg(pv)}
    params = get_settings()[0].model_copy(update=pressures)
    return analyse_tree(
        _prob, start, end, params=params, mm_per_px=mm_per_px, image=_image, lumen=lumen
    )


@st.cache_resource
def get_tree_runner() -> TreeSurrogateRunner | None:
    model_dir = tree_view.tree_model_dir()
    return None if model_dir is None else TreeSurrogateRunner.from_dir(model_dir)


@st.cache_resource
def get_uncertainty():
    return load_uncertainty()


@st.cache_resource(max_entries=32)
def baseline(key: str, _profile, pa: float, pv: float, v_hyp: float):
    params = get_settings()[0].model_copy(update={"v_hyp": v_hyp})
    return core.evaluate(_profile, params, pa, pv, runner=get_models().surrogate)


@st.cache_resource(max_entries=32)
def input_mc(key: str, _profile, pa: float, pv: float, v_hyp: float, mm_per_px, assumed: bool,
             v_range=None):  # fmt: skip
    params = get_settings()[0].model_copy(update={"v_hyp": v_hyp})
    measurement = Measurement(mm_per_px, assumed)
    return core.input_uncertainty(_profile, params, pa, pv, measurement, get_uncertainty(),
                                  v_range)  # fmt: skip


models = get_models()
params, planner_cfg, catalog = get_settings()
ucfg = get_uncertainty()[0]
if models.surrogate is None:
    st.warning("No exported surrogate found (run `vstent export-surrogate`); solver only.")

st.title("Virtual stenting planner")
st.caption(
    "Reduced-order FFR physics + ONNX surrogate. Research prototype, synthetic and "
    "public data only."
)

# --- sidebar: input ------------------------------------------------------------------------
with st.sidebar:
    st.header("Input")
    examples = core.arcade_examples()
    mode = st.radio("Mode", ["Synthetic vessel", "Angiogram image"], key="mode")
    with st.expander("Haemodynamics", expanded=False):
        pa_mmhg = st.number_input("Aortic pressure Pa (mmHg)", 50.0, 150.0, params.pa_mmhg, 1.0)
        pv_mmhg = st.number_input("Venous pressure Pv (mmHg)", 0.0, 20.0, params.pv_mmhg, 0.5)
    pa, pv = mmhg_to_pa(pa_mmhg), mmhg_to_pa(pv_mmhg)

state: core.CaseState | None = None
image_input: ImageInput | None = None
image_input2: ImageInput | None = None  # P5c: second view
example: core.TwoViewExample | None = None
prob = prob2 = qca = two = None
rf = rf2 = None  # P7: the cine runs' analyses
run_t = run2_t = 0
hyperemic = use_measured = False
tree_mode = False
tree_res = tree_post = tree_plans = None
tree_stents: list[Stent] = []
tree_outlet = 0

if mode == "Synthetic vessel":
    with st.sidebar:
        preset = st.selectbox("Vessel", core.PRESETS, key="preset")
        seed = st.number_input("Seed", 0, 10_000, 1, key="seed") if preset == core.PRESETS[2] else 0
    case = core.synthetic_case(preset, int(seed), params)
    case_pa, case_pv = (case.pa, case.pv) if preset == core.PRESETS[2] else (pa, pv)
    case_params = params.model_copy(update={"v_hyp": case.v_hyp})
    case_key = f"syn:{preset}:{seed}:{case_pa:.1f}:{case_pv:.1f}"
    pre = baseline(case_key, case.profile, case_pa, case_pv, case.v_hyp)
    state = core.CaseState(
        "synthetic", case_params, case_pa, case_pv, case.profile, pre, description=case.description
    )
    state.uncertainty = input_mc(case_key, case.profile, case_pa, case_pv, case.v_hyp, None, False)
    if models.surrogate is not None:  # e.g. Pa set outside the surrogate's training range
        state.warnings += range_warnings(case.profile, case_pa, case_pv)
    if preset == core.PRESETS[2]:
        st.sidebar.caption(
            f"Pa {case_pa / 133.322:.0f} mmHg, Pv {case_pv / 133.322:.1f} mmHg and v_hyp "
            f"{case.v_hyp:.2f} m/s come from the synthetic sampler."
        )
else:
    run = None
    with st.sidebar:
        sources = ["Upload"] + (["ARCADE example"] if examples else []) + [TWO_VIEW_SOURCE]
        source = st.radio("Image source", sources, key="source")
        if source == "Upload":
            up = st.file_uploader("PNG, JPG or DICOM (single frame or cine run)",
                                  type=["png", "jpg", "jpeg", "dcm"])  # fmt: skip
            if up is not None:
                data = up.getvalue()
                if is_multiframe_dicom(data, up.name) and models.segmentation is not None:
                    run = read_run_bytes(data, up.name)
                    rf = run_analysis(run.sha256, run)
                    run_t = int(st.number_input(
                        "Frame of the run (0-based)", 0, len(run) - 1, rf.analysis_frame,
                        key=f"cine_{run.sha256[:12]}",
                    ))  # fmt: skip
                    st.caption(core.cine_caption(len(run), rf))
                    image_input = run.frame(run_t)
                else:
                    image_input = read_image(data, up.name)
        elif source == TWO_VIEW_SOURCE:
            example = get_two_view_example()
            image_input, image_input2 = example.views
            st.caption("A synthetic coronary tree rendered in two C-arm views with known "
                       "geometry (not clinical data). Its rendered vessel masks replace the "
                       "segmentation.")  # fmt: skip
        else:
            choice = st.selectbox(
                "ARCADE stenosis/val image (CC0)", [p.name for p in examples], key="example"
            )
            path = next(p for p in examples if p.name == choice)
            image_input = read_image(path.read_bytes(), f"ARCADE stenosis/val {path.name}")
        has_geometry = image_input is not None and image_input.geometry is not None
        second = source == "Upload" and has_geometry and st.toggle(
            "Second view (3D)", key="two_view",
            help="A second DICOM run of the same vessel from another angle (≥ 30° apart): the "
            "vessel is reconstructed in 3D, so lengths are free of foreshortening and diameters "
            "use their true magnification.",
        )  # fmt: skip
        if second:
            up2 = st.file_uploader("Second view: DICOM with projection geometry", type=["dcm"],
                                   key="upload2")  # fmt: skip
            if up2 is not None:
                data2 = up2.getvalue()
                if is_multiframe_dicom(data2, up2.name) and models.segmentation is not None:
                    run2 = read_run_bytes(data2, up2.name)
                    rf2 = run_analysis(run2.sha256, run2)
                    run2_t = rf2.analysis_frame
                    image_input2 = run2.frame(run2_t)
                else:
                    image_input2 = read_image(data2, up2.name)
                if image_input2.geometry is None:
                    st.warning(
                        f"{image_input2.name} has no complete projection geometry; two "
                        "views need DICOM with the positioner angles and distances."
                    )
                    image_input2 = None
        st.subheader("Calibration")
        if image_input2 is not None:
            st.caption("Two views: each diameter is converted at the pixel size of its 3D "
                       "point, from both views' projection geometry.")  # fmt: skip
            mm_per_px = None
        else:
            options = ["Assume 0.30 mm/px", "Enter pixel size"]
            if image_input is not None and image_input.pixel_spacing_mm:
                options.insert(0, DICOM_CALIBRATION)  # magnification-corrected (P5)
            calib = st.radio("Pixel size", options, key="calib")
            mm_per_px = None
            if calib == "Enter pixel size":
                mm_per_px = st.number_input("mm per pixel", 0.05, 1.0, 0.30, 0.01, key="mm_px")
            elif calib == DICOM_CALIBRATION:
                mm_per_px = image_input.pixel_spacing_mm
                st.caption(image_input.note)
            elif image_input is not None and image_input.detector_spacing_mm:
                st.caption(image_input.note)
        if image_input2 is None:
            tree_mode = st.toggle(
                "Side branches (tree model)", key="tree_mode",
                help="Measure the side branches leaving the path and compute FFR on the whole "
                "tree.",
            )  # fmt: skip
        lumen = LUMEN_CHOICES[st.selectbox(
            "Lumen measurement", list(LUMEN_CHOICES), key="lumen",
            help="Intensity edges: where the image brightness changes fastest (they sit inside "
            "the blurred wall, so they read vessels narrow). Size from the imaging chain: the "
            "edges' shape, with the absolute size from the blur and edge enhancement fitted on "
            "healthy stretches (about 1 s). Imaging-chain model: the whole lumen fitted (resolves "
            "tight lesions; several seconds on long vessels).",
        )]  # fmt: skip
        if rf is not None:
            st.subheader("Flow (cine run)")
            hyperemic = st.checkbox(
                "Recorded under hyperemia (adenosine)", key="hyperemic",
                help="DICOM does not record it. A resting run's contrast velocity is converted "
                "with the FAVOR Pilot relation.",
            )  # fmt: skip
            use_measured = st.toggle(
                "Use the measured flow velocity", key="use_measured",
                help="Replace the fixed hyperemic velocity with the one measured from the "
                "contrast's transit (P7). Not yet validated against real flow.",
            )  # fmt: skip
    if image_input is None:
        st.info("Upload an angiogram (or pick an ARCADE example) to start.")
        st.stop()
    if models.segmentation is None and example is None:
        st.error("No exported segmentation model found (run `vstent export-seg`).")
        st.stop()
    if st.session_state.get("image_sha") != image_input.sha256:
        st.session_state.update(image_sha=image_input.sha256, start=None, end=None, last_click=None)
    sha2 = None if image_input2 is None else image_input2.sha256
    if st.session_state.get("image2_sha") != sha2:
        st.session_state.update(image2_sha=sha2, start2=None, end2=None, last_click2=None)
    if example is not None:
        prob, prob2 = example.probs
    else:
        prob = segment(image_input.sha256, image_input.image)
        if image_input2 is not None:
            prob2 = segment(image_input2.sha256, image_input2.image)

# --- stent selection (read before drawing so the image overlay can show it) ------------------
lengths, diameters = catalog.lengths_mm, catalog.diameters_mm

if mode == "Angiogram image":
    img_col, ctl_col = st.columns([3, 2])
    with ctl_col:
        st.subheader("Vessel path")
        st.caption(
            "Click the start (proximal, near the ostium) and then the end point on the "
            "image, or use the buttons / manual entry."
        )
        b1, b2 = st.columns(2)
        if b1.button("Auto path (longest)", key="auto"):
            a, b = longest_path_endpoints(prob >= THRESHOLD)
            st.session_state.update(start=tuple(map(int, a)), end=tuple(map(int, b)))
        if b2.button("Reset points", key="reset"):
            st.session_state.update(start=None, end=None)
        with st.form("manual_points"):
            h, w = image_input.image.shape
            c1, c2, c3, c4 = st.columns(4)
            sr = c1.number_input("start row", 0, h - 1, (st.session_state.start or (h // 2, 0))[0])
            sc = c2.number_input("start col", 0, w - 1, (st.session_state.start or (0, w // 2))[1])
            er = c3.number_input("end row", 0, h - 1, (st.session_state.end or (h // 2, 0))[0])
            ec = c4.number_input("end col", 0, w - 1, (st.session_state.end or (0, w // 2))[1])
            if st.form_submit_button("Set points"):
                st.session_state.update(start=(int(sr), int(sc)), end=(int(er), int(ec)))
        if image_input2 is not None:
            st.caption(
                "View 2: click the same two landmarks on the second image. The grey lines show "
                "where they must lie (the epipolar lines of the view-1 points)."
            )
            if example is not None and st.button("Use the example's landmarks", key="marks"):
                (a1, b1), (a2, b2) = example.landmarks
                st.session_state.update(start=a1, end=b1, start2=a2, end2=b2)
            with st.form("manual_points2"):
                h2, w2 = image_input2.image.shape
                d1, d2, d3, d4 = st.columns(4)
                s2 = st.session_state.get("start2") or (h2 // 2, 0)
                e2 = st.session_state.get("end2") or (h2 // 2, 0)
                sr2 = d1.number_input("start row 2", 0, h2 - 1, s2[0])
                sc2 = d2.number_input("start col 2", 0, w2 - 1, s2[1])
                er2 = d3.number_input("end row 2", 0, h2 - 1, e2[0])
                ec2 = d4.number_input("end col 2", 0, w2 - 1, e2[1])
                if st.form_submit_button("Set view-2 points"):
                    st.session_state.update(start2=(int(sr2), int(sc2)), end2=(int(er2), int(ec2)))
    start, end = st.session_state.get("start"), st.session_state.get("end")
    start2, end2 = st.session_state.get("start2"), st.session_state.get("end2")
    if image_input2 is not None:
        if start and end and start2 and end2:
            names = (image_input.name, image_input2.name)
            two_key = (f"two:{image_input.sha256}:{image_input2.sha256}:{start}:{end}:{start2}:"
                       f"{end2}:{pa:.1f}:{pv:.1f}:{lumen}")  # fmt: skip
            try:
                two = analyse_two(
                    two_key, (start, start2), (end, end2), pa, pv, lumen, (prob, prob2),
                    (image_input.image, image_input2.image),
                    (image_input.geometry, image_input2.geometry),
                )  # fmt: skip
            except ValueError as exc:
                st.error(f"Could not reconstruct the vessel from the two views: {exc}")
            fv = None
            if two is not None and rf is not None and rf2 is not None:
                rec = two.reconstruction
                fv = combine_runs([
                    run_velocity(f"{two_key}:{k}:{hyperemic}", r, two.qca[k].centerline, ft,
                                 None, hyperemic, rec.view_s[:, k], rec.s)[0]
                    for k, (r, ft) in enumerate(((rf, run_t), (rf2, run2_t)))
                ])  # fmt: skip
                st.sidebar.caption("Both runs, 3D path: " + core.flow_caption(fv))
                if use_measured and fv.accepted:
                    two = analyse_two(
                        two_key, (start, start2), (end, end2), pa, pv, lumen, (prob, prob2),
                        (image_input.image, image_input2.image),
                        (image_input.geometry, image_input2.geometry), v_hyp=fv.v_hyp,
                    )  # fmt: skip
            if two is not None:
                used = fv is not None and use_measured and fv.accepted
                case_params = params.model_copy(update={"v_hyp": fv.v_hyp}) if used else params
                v_range = fv.v_hyp_interval if used else None
                pre = baseline(two_key, two.profile, pa, pv, case_params.v_hyp)
                state = core.CaseState(
                    "image", case_params, pa, pv, two.profile, pre, image=image_input,
                    points=[start, end], description=" + ".join(names),
                    warnings=list(two.warnings), two_view=two, image2=image_input2,
                    points2=[start2, end2],
                )  # fmt: skip
                if fv is not None:
                    state.flow = core.flow_dict(fv, used, rf.ed)
                iso_mm = image_input.geometry.iso_pixel_m * 1e3
                state.uncertainty = input_mc(two_key, two.profile, pa, pv, case_params.v_hyp,
                                             iso_mm, False, v_range)  # fmt: skip
    elif start and end and tree_mode:
        try:
            tree_res = analyse_tree_cached(
                image_input.sha256, start, end, mm_per_px, pa, pv, lumen, prob, image_input.image
            )
        except ValueError as exc:
            st.error(f"Could not trace the vessel tree: {exc}")
    elif start and end:
        try:
            result = analyse(
                image_input.sha256, start, end, mm_per_px, pa, pv, lumen, prob, image_input.image
            )
        except ValueError as exc:
            st.error(f"Could not trace the vessel: {exc}")
            result = None
        fv = None
        if result is not None and rf is not None:
            q0 = result.qca
            fv_key = f"flow:{image_input.sha256}:{start}:{end}:{mm_per_px}:{run_t}:{hyperemic}"
            calibrated = None if q0.calibration_assumed else q0.mm_per_px
            fv, tr = run_velocity(fv_key, rf, q0.centerline, run_t, calibrated, hyperemic)
            st.sidebar.caption(core.flow_caption(fv, tr))
            if use_measured and fv.accepted:
                result = analyse(image_input.sha256, start, end, mm_per_px, pa, pv, lumen, prob,
                                 image_input.image, v_hyp=fv.v_hyp)  # fmt: skip
        if result is not None:
            qca = result.qca
            used = fv is not None and use_measured and fv.accepted
            case_params = params.model_copy(update={"v_hyp": fv.v_hyp}) if used else params
            img_key = (f"img:{image_input.sha256}:{start}:{end}:{mm_per_px}:{pa:.1f}:{pv:.1f}:"
                       f"{case_params.v_hyp:.4f}")  # fmt: skip
            pre = baseline(img_key, qca.profile, pa, pv, case_params.v_hyp)
            state = core.CaseState(
                "image",
                case_params,
                pa,
                pv,
                qca.profile,
                pre,
                qca=qca,
                image=image_input,
                points=[start, end],
                description=image_input.name,
                warnings=list(result.warnings),
            )
            if fv is not None:
                state.flow = core.flow_dict(fv, used, rf.ed)
            state.uncertainty = input_mc(
                img_key, qca.profile, pa, pv, case_params.v_hyp, qca.mm_per_px,
                qca.calibration_assumed, fv.v_hyp_interval if used else None,
            )  # fmt: skip

# --- stents: manual controls or a selected plan ---------------------------------------------
stents: list[Stent] = []
show = "Manual stent"
if state is not None:
    # A different vessel (not just different pressures) starts with no stent and no plan shown,
    # instead of carrying the previous case's controls over. The geometry hash makes a new pixel
    # size a new vessel (its length and diameters change).
    geometry = hashlib.sha1(state.profile.s.tobytes() + state.profile.r.tobytes()).hexdigest()
    vessel_id = f"{mode}:{state.description}:{state.points}:{geometry}"
    if st.session_state.get("vessel_id") != vessel_id:
        st.session_state.vessel_id = vessel_id
        st.session_state.place = False
        st.session_state.pop("stent_start", None)
        st.session_state.pop("show", None)
    length_mm_total = m_to_mm(state.profile.length)
    plan_key = f"{vessel_id}:{state.pa:.1f}:{state.pv:.1f}:{state.params.v_hyp:.4f}"
    plans = st.session_state.get("plans") if st.session_state.get("plan_key") == plan_key else None
    with st.sidebar:
        st.header("Virtual stent")
        place = st.toggle("Place a stent", key="place")
        s_len = st.selectbox(
            "Length (mm)", lengths, index=lengths.index(18) if 18 in lengths else 0, key="stent_len"
        )
        s_dia = st.selectbox(
            "Diameter (mm)",
            diameters,
            index=diameters.index(3.0) if 3.0 in diameters else 0,
            key="stent_dia",
        )
        max_start = max(0.0, float(np.floor((length_mm_total - s_len) * 2) / 2))
        # Initialise / clamp through session state only (no ``value=``): the vessel or the
        # stent length may have changed since the last run.
        if "stent_start" not in st.session_state:
            mld_mm = m_to_mm(state.profile.s[state.profile.mld()[1]])
            st.session_state.stent_start = min(max_start, max(0.0, round(mld_mm - s_len / 2)))
        elif st.session_state.stent_start > max_start:
            st.session_state.stent_start = max_start
        s_start = st.slider(
            "Start (mm from the path start)",
            0.0,
            max(max_start, 0.5),
            step=0.5,
            key="stent_start",
            disabled=s_len > length_mm_total,
        )
        if plans is not None and plans.plans:
            show = st.radio(
                "Show on plots",
                ["Manual stent"] + [f"Plan {k}" for k in range(1, len(plans.plans) + 1)],
                key="show",
            )
    if show != "Manual stent" and plans is not None:
        stents = list(plans.plans[int(show.split()[1]) - 1].stents)
    elif place and s_len <= length_mm_total:
        stents = [Stent(mm_to_m(s_start), mm_to_m(s_len), mm_to_m(s_dia))]
    state.pre_assessment = core.assess(
        state.pre, state.uncertainty.pre_samples, ucfg.pre_pci_threshold, ucfg.level
    )
    state.edge = mm_to_m(planner_cfg.edge_mm)  # same stent edges as the planner
    if stents:
        stented, post = core.stent_update(
            state.profile, stents, state.pre, state.params, state.pa, state.pv, models.surrogate,
            edge=state.edge,
        )  # fmt: skip
        state.stents, state.post = stents, post
        samples = state.uncertainty.post_samples(stents, state.edge)
        state.post_assessment = core.assess(post, samples, planner_cfg.target_ffr, ucfg.level)
        state.warnings += core.stent_warnings(state.profile, stents, planner_cfg.max_diameter_ratio)
    else:
        stented = None
    state.plans = plans
    if plans is not None:
        state.plan_assessments = plan_assessments(plans, state.uncertainty, ucfg.level)

# --- tree mode: path, stent and plan controls (read before drawing the overlay) -------------
if tree_res is not None:
    tree = tree_res.qca.tree
    tree_key = (
        f"{image_input.sha256}:{st.session_state.get('start')}:{st.session_state.get('end')}:"
        f"{mm_per_px}:{pa:.1f}:{pv:.1f}"
    )
    if st.session_state.get("tree_key") != tree_key:  # a new tree starts without stent / plans
        st.session_state.tree_key = tree_key
        for key in ("tree_place", "tree_start", "tree_show", "tree_plans"):
            st.session_state.pop(key, None)
    with st.sidebar:
        st.header("Virtual stent (tree)")
        tree_outlet = st.selectbox(
            "Path to", range(tree.n_branches), format_func=lambda k: tree.names[k],
            key="tree_outlet",
        )  # fmt: skip
        path_mm = m_to_mm(tree.path_length(tree_outlet))
        place = st.toggle("Place a stent on this path", key="tree_place")
        t_len = st.selectbox("Length (mm)", lengths, index=lengths.index(18) if 18 in lengths
                             else 0, key="tree_len")  # fmt: skip
        t_dia = st.selectbox("Diameter (mm)", diameters, index=diameters.index(3.0)
                             if 3.0 in diameters else 0, key="tree_dia")  # fmt: skip
        max_start = max(0.0, float(np.floor((path_mm - t_len) * 2) / 2))
        if st.session_state.get("tree_start", 0.0) > max_start:
            st.session_state.tree_start = max_start
        t_start = st.slider("Start (mm along the path)", 0.0, max(max_start, 0.5), step=0.5,
                            key="tree_start", disabled=t_len > path_mm)  # fmt: skip
        stored = st.session_state.get("tree_plans")
        tree_plans = stored if stored is not None and stored.outlet == tree_outlet else None
        show = "Manual stent"
        if tree_plans is not None and tree_plans.plans:
            show = st.radio(
                "Show on plots",
                ["Manual stent"] + [f"Plan {k}" for k in range(1, len(tree_plans.plans) + 1)],
                key="tree_show",
            )
    if show != "Manual stent" and tree_plans is not None:
        tree_stents = list(tree_plans.plans[int(show.split()[1]) - 1].stents)
    elif place and t_len <= path_mm:
        tree_stents = [Stent(mm_to_m(t_start), mm_to_m(t_len), mm_to_m(t_dia))]
    if tree_stents:
        stented = apply_path_stents(
            tree, tree_outlet, tree_stents, edge=mm_to_m(planner_cfg.edge_mm)
        )
        tree_post = solve_tree(stented, tree_res.params, r_outlet=tree_res.result.r_outlet)

# --- image panel --------------------------------------------------------------------------------
if mode == "Angiogram image":
    with img_col:
        points = [p for p in (st.session_state.get("start"), st.session_state.get("end")) if p]
        if tree_res is not None:
            overlay = tree_view.tree_overlay(
                image_input.image, prob, tree_res.qca, tree_outlet, tree_stents, points
            )
        elif image_input2 is not None:
            overlay = core.two_view_overlay(image_input.image, prob, 0, two, stents, points)
        else:
            overlay = core.overlay_image(image_input.image, prob, qca, stents, points)
        _click_points(overlay, image_input.image.shape, f"click_{image_input.sha256[:12]}",
                      ("start", "end"), "last_click")  # fmt: skip
        st.caption(
            f"{image_input.name}: blue = segmented vessel (p ≥ {THRESHOLD}), yellow = "
            "centreline, green dot = start, orange = end / stent."
        )
        if image_input2 is not None:
            g1, g2 = image_input.geometry, image_input2.geometry
            guides = [core.epipolar_guide(g1, p, g2) for p in points]
            points2 = [p for p in (start2, end2) if p]
            overlay2 = core.two_view_overlay(image_input2.image, prob2, 1, two, stents, points2,
                                             guides)  # fmt: skip
            _click_points(overlay2, image_input2.image.shape, f"click2_{image_input2.sha256[:12]}",
                          ("start2", "end2"), "last_click2")  # fmt: skip
            st.caption(
                f"{image_input2.name}: the same landmarks; grey = epipolar lines of the view-1 "
                "points, yellow = the 3D centreline projected back into each view."
            )
    if tree_mode and tree_res is None:
        st.info("Set a start and an end point to analyse the vessel tree.")
        st.stop()
    elif image_input2 is not None and state is None:
        st.info("Set the start and end points in both views to reconstruct the vessel in 3D.")
        st.stop()
    elif not tree_mode and state is None:
        st.info("Set a start and an end point to analyse the vessel.")
        st.stop()

# --- tree mode: results, planner, export ------------------------------------------------------
if tree_res is not None:
    tree, pre_t = tree_res.qca.tree, tree_res.result
    st.subheader("Vessel tree")
    m = st.columns(4)
    m[0].metric("Branches modelled", tree.n_branches)
    main_post = None if tree_post is None else float(tree_post.ffr[0][-1])
    m[1].metric(
        "Main-path distal FFR (tree)", f"{pre_t.ffr[0][-1]:.3f}",
        None if main_post is None else f"{main_post - pre_t.ffr[0][-1]:+.3f}",
    )  # fmt: skip
    m[2].metric("Main path as a single vessel", f"{tree_res.single.ffr[-1]:.3f}")
    m[3].metric(f"FFR at {tree.names[tree_outlet]}", f"{pre_t.ffr[tree_outlet][-1]:.3f}")
    runner = get_tree_runner()
    sur = None
    if runner is not None:
        try:
            sur = runner.predict_tree(tree, pre_t.pa, pre_t.pv, pre_t.r_outlet)[1]
            for w in floor_warnings(sur, tree.names, pre_t.pa, pre_t.pv):
                st.warning(w)
        except ValueError as exc:
            st.caption(f"Tree surrogate not used: {exc}")
    st.dataframe(tree_view.branch_table(tree_res, tree_post, sur), hide_index=True)
    st.plotly_chart(tree_view.tree_ffr_figure(tree_res, tree_outlet, tree_post))
    for w in tree_res.warnings:
        st.warning(w)
    st.caption(tree_view.TREE_NOTE)
    if st.button(f"Suggest plan on the path to {tree.names[tree_outlet]}", key="tree_suggest"):
        st.session_state.tree_plans = plan_tree_stents(
            tree, tree_outlet, catalog=catalog, config=planner_cfg, params=tree_res.params,
            pa=pre_t.pa, pv=pre_t.pv,
        )  # fmt: skip
        st.session_state.pop("tree_show", None)
        st.rerun()
    if tree_plans is not None:
        st.write(
            f"{tree_plans.message} (tree {tree_plans.backend}; {tree_plans.n_feasible} feasible "
            f"of {tree_plans.n_candidates} candidates, {tree_plans.n_pairs} pairs; "
            f"{tree_plans.seconds:.2f} s)"
        )
        if tree_plans.plans:
            st.dataframe(
                [
                    {
                        "Plan": k,
                        "Stents (mm along the path)": " + ".join(
                            f"{m_to_mm(s.diameter):.2f} × {m_to_mm(s.length):.0f} mm "
                            f"@ {m_to_mm(s.s_start):.1f} mm" for s in p.stents
                        ),
                        "Total length (mm)": round(m_to_mm(p.total_length), 1),
                        f"FFR at {tree.names[tree_outlet]}": round(p.ffr, 3),
                        "Meets target": p.meets_target,
                        **{f"FFR {nm}": round(f, 3)
                           for nm, f in zip(tree.names, p.ffr_outlets, strict=True)},
                    }
                    for k, p in enumerate(tree_plans.plans, 1)
                ],
                hide_index=True,
            )  # fmt: skip
    st.download_button(
        "Download tree case report (JSON)",
        tree_view.tree_report_json(tree_res, tree_outlet, tree_stents, tree_post, tree_plans,
                                   image_input.name),
        file_name="vstent_tree_case.json", mime="application/json", key="tree_report",
    )  # fmt: skip
    st.stop()

# --- results ----------------------------------------------------------------------------------
p, pre, post = state.profile, state.pre, state.post
st.subheader("Results")
m = st.columns(5)
length_label = "3D path length" if state.two_view else "Path length"
m[0].metric(length_label, f"{m_to_mm(p.length):.1f} mm")
if state.two_view is not None:
    st.caption(core.two_view_caption(state.two_view))
m[1].metric("MLD", f"{m_to_mm(p.mld()[0]):.2f} mm")
m[2].metric("Max DS", f"{p.percent_stenosis().max():.0f}%")
m[3].metric(
    "Distal FFR (solver)",
    f"{pre.distal_solver:.3f}",
    None if post is None else f"{post.distal_solver - pre.distal_solver:+.3f} after stent",
)
m[4].metric(
    "Distal FFR (surrogate)",
    "n/a" if pre.surrogate is None else f"{pre.distal_surrogate:.3f}",
    None
    if post is None or post.surrogate is None
    else f"{post.distal_surrogate - pre.distal_surrogate:+.3f} after stent",
)
if post is not None:
    st.success(
        f"Post-PCI distal FFR: solver **{post.distal_solver:.3f}**"
        + ("" if post.surrogate is None else f", surrogate **{post.distal_surrogate:.3f}**")
        + " — "
        + ", ".join(
            f"{m_to_mm(s.diameter):.2f} × {m_to_mm(s.length):.0f} mm at {m_to_mm(s.s_start):.1f} mm"
            for s in stents
        )
    )
lines = []
for label, a in (("Pre-PCI", state.pre_assessment), ("Post-PCI", state.post_assessment)):
    if a is None:
        continue
    parts = [f"inputs **{a.inputs.text()}** (solver)"] if a.inputs else []
    parts += [f"model **{a.model.text()}** (surrogate)"] if a.model else []
    verdict = "**grey zone**" if a.flagged else "outside the grey zone"
    lines.append(
        f"{label} distal FFR, {a.level_text} intervals: {', '.join(parts)}; "
        f"threshold {a.threshold:.2f}: {verdict}"
    )
st.markdown("  \n".join(lines))
for label, a in (("Pre-PCI", state.pre_assessment), ("Post-PCI", state.post_assessment)):
    for reason in a.reasons(f"{label} distal FFR") if a is not None else []:
        st.warning(f"Grey zone: {reason}.")
mc = state.uncertainty
st.caption(
    f"Input assumptions varied: {', '.join(SOURCE_LABELS[s] for s in mc.sources)} "
    f"({mc.n_samples} Monte Carlo draws through the solver). {core.UNCERTAINTY_NOTE}"
    + ("" if pre.model_interval else " No calibrated ensemble loaded: no model interval.")
)
for w in state.warnings:
    st.warning(w)
if qca is not None and qca.calibration_assumed:
    st.caption("Pixel size assumed (0.30 mm/px): absolute sizes and FFR are illustrative.")

c1, c2 = st.columns(2)
c1.plotly_chart(
    core.profile_figure(p, stented, stents, qca.measured_diameter_mm if qca is not None else None),
    key="profile_fig",
)
c2.plotly_chart(
    core.ffr_figure(pre, post, (state.pre_assessment, state.post_assessment)), key="ffr_fig"
)

# --- planner ------------------------------------------------------------------------------------
st.subheader("Stent plan suggestion")
st.caption(
    f"Shortest plan reaching distal FFR ≥ {planner_cfg.target_ffr:.2f}; generic catalogue "
    f"({min(diameters):g}–{max(diameters):g} mm × {min(lengths):g}–{max(lengths):g} mm), "
    f"diameter ≤ {planner_cfg.max_diameter_ratio} × distal reference, landing zones DS < "
    f"{planner_cfg.landing_max_ds_pct:g}%, up to {planner_cfg.max_stents} stents. "
    f"Grey zone: an interval of the plan's post-PCI FFR contains {planner_cfg.target_ffr:.2f}."
    + (
        " Lengths are measured in one view: foreshortening makes them shorter than the vessel "
        "(two views: `vstent analyse --view2`)."
        if image_input is not None and image_input2 is None
        else ""
    )
)
if st.button("Suggest plan", key="suggest"):
    backend = "surrogate" if models.surrogate is not None else "solver"
    st.session_state.plans = plan_stents(
        p,
        catalog=catalog,
        config=planner_cfg.model_copy(update={"backend": backend}),
        params=state.params,
        runner=models.surrogate,
        pa=state.pa,
        pv=state.pv,
    )
    st.session_state.plan_key = plan_key
    st.rerun()
if state.plans is not None:
    pr = state.plans
    st.write(
        f"{pr.message} ({pr.backend} backend; {pr.n_feasible} feasible of "
        f"{pr.n_candidates} candidates, {pr.n_pairs} pairs; {pr.seconds:.2f} s)"
    )
    if pr.plans:
        model_level = next((a.model.level for a in state.plan_assessments if a.model), ucfg.level)
        st.dataframe(
            [
                {
                    "Plan": k,
                    "Stents": " + ".join(
                        f"{m_to_mm(s.diameter):.2f} × {m_to_mm(s.length):.0f} mm "
                        f"@ {m_to_mm(s.s_start):.1f} mm"
                        for s in pl.stents
                    ),
                    "Total length (mm)": round(m_to_mm(pl.total_length), 1),
                    f"Post FFR ({pr.backend})": round(pl.ffr, 3),
                    "Solver check": None if pl.ffr_solver is None else round(pl.ffr_solver, 3),
                    "Meets target": pl.meets_target,
                    f"Model {model_level:.0%}": a.model.text() if a.model else None,
                    f"Inputs {ucfg.level:.0%}": a.inputs.text() if a.inputs else None,
                    "Grey zone": a.flagged,
                }
                for k, (pl, a) in enumerate(zip(pr.plans, state.plan_assessments, strict=True), 1)
            ],
            hide_index=True,
        )
        with st.expander("Pareto front (total length vs post-PCI FFR)"):
            import plotly.graph_objects as go

            fig = go.Figure(
                go.Scatter(
                    x=[m_to_mm(q.total_length) for q in pr.pareto],
                    y=[q.ffr for q in pr.pareto],
                    mode="lines+markers",
                )
            )
            fig.add_hline(y=planner_cfg.target_ffr, line_dash="dot")
            fig.update_layout(
                template="plotly_white",
                height=300,
                xaxis_title="Total stent length (mm)",
                yaxis_title="Post-PCI distal FFR",
            )
            st.plotly_chart(fig, key="pareto_fig")
        st.caption("Pick a plan under **Show on plots** in the sidebar to draw it.")

# --- export -------------------------------------------------------------------------------------
st.subheader("Case report")
signature = f"{plan_key}:{[(s.s_start, s.length, s.diameter) for s in stents]}"
if st.button("Prepare case report", key="prepare"):
    st.session_state.report = (core.report_json(state, models), core.report_png(state), signature)
prepared = st.session_state.get("report")
if prepared is not None and prepared[2] == signature:
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download report (JSON)",
        prepared[0],
        file_name="vstent_case_report.json",
        mime="application/json",
        key="dl_json",
    )
    d2.download_button(
        "Download plots (PNG)",
        prepared[1],
        file_name="vstent_case_plots.png",
        mime="image/png",
        key="dl_png",
    )
st.caption(
    "The report lists the disclaimer and every assumption (calibration, flow model, "
    "model versions); it contains no image pixels."
)
st.caption(f"Server time for this update: {(time.perf_counter() - _T0) * 1e3:.0f} ms")
