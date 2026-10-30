"""vstent Streamlit app (``vstent app``). Research use only — not for clinical decisions.

Synthetic mode: a preset or random synthetic vessel. Image mode: upload a PNG / JPG / DICOM
(or pick an ARCADE example), see the segmentation, click start and end points, set the
calibration. Then: diameter and FFR plots (solver and surrogate), a virtual stent with instant
post-PCI FFR, the plan optimiser, and a case report export. Distal FFR comes with 90% intervals
(surrogate ensemble: conformal; inputs: Monte Carlo through the solver) and grey-zone flags.
Slow steps (segmentation, QCA, the pre-PCI Monte Carlo) are cached so slider updates only
recompute the stent, its post-PCI draws and the plots.
"""

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

from vstent.app import core  # noqa: E402
from vstent.imaging.centerline import longest_path_endpoints  # noqa: E402
from vstent.imaging.segment import THRESHOLD  # noqa: E402
from vstent.inference.pipeline import analyse_vessel  # noqa: E402
from vstent.planning.optimizer import plan_stents  # noqa: E402
from vstent.planning.stent import Stent  # noqa: E402
from vstent.units import m_to_mm, mm_to_m, mmhg_to_pa  # noqa: E402

DISPLAY_WIDTH = 512

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


@st.cache_resource(max_entries=32, show_spinner="Measuring the vessel (QCA)…")
def analyse(sha: str, start, end, mm_per_px, pa, pv, _prob):
    params = get_settings()[0]
    return analyse_vessel(
        _prob, start, end, params=params, surrogate=get_models().surrogate, mm_per_px=mm_per_px
    )


@st.cache_resource
def get_uncertainty():
    return core.load_uncertainty()


@st.cache_resource(max_entries=32)
def baseline(key: str, _profile, pa: float, pv: float, v_hyp: float):
    params = get_settings()[0].model_copy(update={"v_hyp": v_hyp})
    return core.evaluate(_profile, params, pa, pv, runner=get_models().surrogate)


@st.cache_resource(max_entries=32)
def input_mc(key: str, _profile, pa: float, pv: float, v_hyp: float, mm_per_px, assumed: bool):
    params = get_settings()[0].model_copy(update={"v_hyp": v_hyp})
    measurement = core.Measurement(mm_per_px, assumed)
    return core.input_uncertainty(_profile, params, pa, pv, measurement, get_uncertainty())


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
image_input: core.ImageInput | None = None
prob = qca = None

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
    if preset == core.PRESETS[2]:
        st.sidebar.caption(
            f"Pa {case_pa / 133.322:.0f} mmHg, Pv {case_pv / 133.322:.1f} mmHg and v_hyp "
            f"{case.v_hyp:.2f} m/s come from the synthetic sampler."
        )
else:
    with st.sidebar:
        sources = ["Upload"] + (["ARCADE example"] if examples else [])
        source = st.radio("Image source", sources, key="source")
        if source == "Upload":
            up = st.file_uploader("PNG, JPG or DICOM", type=["png", "jpg", "jpeg", "dcm"])
            if up is not None:
                image_input = core.read_image(up.getvalue(), up.name)
        else:
            choice = st.selectbox(
                "ARCADE stenosis/val image (CC0)", [p.name for p in examples], key="example"
            )
            path = next(p for p in examples if p.name == choice)
            image_input = core.read_image(path.read_bytes(), f"ARCADE stenosis/val {path.name}")
        st.subheader("Calibration")
        options = ["Assume 0.30 mm/px", "Enter pixel size"]
        if image_input is not None and image_input.pixel_spacing_mm:
            options.append("DICOM ImagerPixelSpacing")
        calib = st.radio("Pixel size", options, key="calib")
        mm_per_px = None
        if calib == "Enter pixel size":
            mm_per_px = st.number_input("mm per pixel", 0.05, 1.0, 0.30, 0.01, key="mm_px")
        elif calib == "DICOM ImagerPixelSpacing":
            mm_per_px = image_input.pixel_spacing_mm
            st.caption(f"{mm_per_px:.4f} mm/px. {image_input.note}")
    if image_input is None:
        st.info("Upload an angiogram (or pick an ARCADE example) to start.")
        st.stop()
    if models.segmentation is None:
        st.error("No exported segmentation model found (run `vstent export-seg`).")
        st.stop()
    if st.session_state.get("image_sha") != image_input.sha256:
        st.session_state.update(image_sha=image_input.sha256, start=None, end=None, last_click=None)
    prob = segment(image_input.sha256, image_input.image)

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
    start, end = st.session_state.get("start"), st.session_state.get("end")
    if start and end:
        try:
            result = analyse(image_input.sha256, start, end, mm_per_px, pa, pv, prob)
        except ValueError as exc:
            st.error(f"Could not trace the vessel: {exc}")
            result = None
        if result is not None:
            qca = result.qca
            img_key = f"img:{image_input.sha256}:{start}:{end}:{mm_per_px}:{pa:.1f}:{pv:.1f}"
            pre = baseline(img_key, qca.profile, pa, pv, params.v_hyp)
            state = core.CaseState(
                "image",
                params,
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
            state.uncertainty = input_mc(
                img_key, qca.profile, pa, pv, params.v_hyp, qca.mm_per_px, qca.calibration_assumed
            )

# --- stents: manual controls or a selected plan ---------------------------------------------
stents: list[Stent] = []
show = "Manual stent"
if state is not None:
    # A different vessel (not just different pressures) starts with no stent and no plan shown,
    # instead of carrying the previous case's controls over.
    vessel_id = f"{mode}:{state.description}:{state.points}"
    if st.session_state.get("vessel_id") != vessel_id:
        st.session_state.vessel_id = vessel_id
        st.session_state.place = False
        st.session_state.pop("stent_start", None)
        st.session_state.pop("show", None)
    length_mm_total = m_to_mm(state.profile.length)
    plan_key = f"{mode}:{state.description}:{state.points}:{state.pa:.1f}"
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
    if stents:
        stented, post = core.stent_update(
            state.profile, stents, state.pre, state.params, state.pa, state.pv, models.surrogate
        )
        state.stents, state.post = stents, post
        state.post_assessment = core.assess(
            post, state.uncertainty.post_samples(stents), planner_cfg.target_ffr, ucfg.level
        )
        state.warnings += core.stent_warnings(state.profile, stents, planner_cfg.max_diameter_ratio)
    else:
        stented = None
    state.plans = plans
    if plans is not None:
        state.plan_assessments = core.plan_assessments(plans, state.uncertainty, ucfg.level)

# --- image panel --------------------------------------------------------------------------------
if mode == "Angiogram image":
    with img_col:
        overlay = core.overlay_image(
            image_input.image,
            prob,
            qca,
            stents,
            [p for p in (st.session_state.get("start"), st.session_state.get("end")) if p],
        )
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates

            click = streamlit_image_coordinates(
                overlay, width=DISPLAY_WIDTH, key=f"click_{image_input.sha256[:12]}"
            )
        except Exception:  # component unavailable (e.g. in tests): static image
            st.image(overlay, width=DISPLAY_WIDTH)
            click = None
        if click and click != st.session_state.get("last_click"):
            st.session_state.last_click = click
            h, w = image_input.image.shape
            pt = (
                int(round(click["y"] * h / click["height"])),
                int(round(click["x"] * w / click["width"])),
            )
            if st.session_state.get("start") is None or st.session_state.get("end") is not None:
                st.session_state.update(start=pt, end=None)
            else:
                st.session_state.end = pt
            st.rerun()
        st.caption(
            f"{image_input.name}: blue = segmented vessel (p ≥ {THRESHOLD}), yellow = "
            "centreline, green dot = start, orange = end / stent."
        )
    if state is None:
        st.info("Set a start and an end point to analyse the vessel.")
        st.stop()

# --- results ----------------------------------------------------------------------------------
p, pre, post = state.profile, state.pre, state.post
st.subheader("Results")
m = st.columns(5)
m[0].metric("Path length", f"{m_to_mm(p.length):.1f} mm")
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
        f"{label} distal FFR, {ucfg.level:.0%} intervals: {', '.join(parts)}; "
        f"threshold {a.threshold:.2f}: {verdict}"
    )
st.markdown("  \n".join(lines))
for label, a in (("Pre-PCI", state.pre_assessment), ("Post-PCI", state.post_assessment)):
    for reason in a.reasons(f"{label} distal FFR") if a is not None else []:
        st.warning(f"Grey zone: {reason}.")
mc = state.uncertainty
st.caption(
    f"Input assumptions varied: {', '.join(core.SOURCE_LABELS[s] for s in mc.sources)} "
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
                    f"Model {ucfg.level:.0%}": a.model.text() if a.model else None,
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
