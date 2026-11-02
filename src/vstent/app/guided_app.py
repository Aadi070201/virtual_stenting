"""vstent guided app: one cine run, step by step (research use only).

Launched by ``vstent app``; the full research view (synthetic vessels, single images, two
views, trees, every interval) is ``vstent app --advanced``. The steps and their logic are in
:mod:`vstent.app.guided`.
"""

import hashlib
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas  # noqa: E402, F401  (imported before the first chart; see streamlit_app.py)
import pyarrow  # noqa: E402, F401
import streamlit as st  # noqa: E402

from vstent import DISCLAIMER  # noqa: E402
from vstent.app import core, guided  # noqa: E402
from vstent.imaging.image_io import is_multiframe_dicom, read_run_bytes  # noqa: E402
from vstent.imaging.run_analysis import analyse_frames  # noqa: E402
from vstent.inference.consensus import run_consensus  # noqa: E402
from vstent.inference.pipeline import analyse_vessel  # noqa: E402
from vstent.planning.optimizer import plan_stents  # noqa: E402
from vstent.planning.stent import Stent  # noqa: E402
from vstent.uncertainty.greyzone import load_uncertainty  # noqa: E402
from vstent.uncertainty.inputs import Measurement  # noqa: E402
from vstent.units import m_to_mm, mm_to_m  # noqa: E402

DISPLAY_WIDTH = 560
st.set_page_config(page_title="vstent: virtual FFR", page_icon="🫀", layout="wide")


# --- cached work --------------------------------------------------------------------------------
@st.cache_resource
def get_models() -> core.Models:
    return core.load_models()


@st.cache_resource
def get_settings():
    return core.load_settings()


@st.cache_resource
def get_uncertainty():
    return load_uncertainty()


@st.cache_resource(max_entries=2, show_spinner="Preparing the demo run…")
def load_demo(kind: str):
    return guided.demo_run(kind)


@st.cache_resource(max_entries=3, show_spinner="Reading every frame: contrast, heartbeats, best "
                   "frame (about a minute)…")  # fmt: skip
def run_frames(sha: str, _run):
    return analyse_frames(_run, get_models().segmentation.predict)


@st.cache_resource(max_entries=16, show_spinner="Finding the vessels…")
def segment(key: str, _image: np.ndarray) -> np.ndarray:
    return get_models().segmentation.predict(_image)


def _params(pa_mmhg: float):
    return get_settings()[0].model_copy(update={"pa_mmhg": pa_mmhg})


@st.cache_resource(max_entries=32, show_spinner="Measuring the vessel…")
def measure(key: str, _prob, _image, start, end, mm_per_px, pa_mmhg):
    return analyse_vessel(_prob, start, end, params=_params(pa_mmhg), mm_per_px=mm_per_px,
                          image=_image)  # fmt: skip


@st.cache_resource(max_entries=16, show_spinner="Measuring the vessel on the other heartbeats…")
def heartbeats(key: str, _rf, frame: int, start, end, mm_per_px, pa_mmhg, _reference):
    params = _params(pa_mmhg)

    def one(p, im, a, b):
        return analyse_vessel(p, a, b, params=params, mm_per_px=mm_per_px, image=im)

    return run_consensus(_rf, frame, start, end, get_models().segmentation.predict, one,
                         reference_analysis=_reference)  # fmt: skip


@st.cache_resource(max_entries=16, show_spinner="Estimating how certain the result is…")
def uncertainty(key: str, _profile, pa_mmhg: float, mm_per_px: float, assumed: bool):
    p = _params(pa_mmhg)
    return core.input_uncertainty(_profile, p, p.pa, p.pv, Measurement(mm_per_px, assumed),
                                  get_uncertainty())  # fmt: skip


@st.cache_resource(max_entries=16, show_spinner="Trying stent lengths, diameters and positions…")
def suggest(key: str, _profile, pa_mmhg: float):
    p, cfg, catalog = _params(pa_mmhg), get_settings()[1], get_settings()[2]
    runner = get_models().surrogate
    if runner is None:
        cfg = cfg.model_copy(update={"backend": "solver"})
    return plan_stents(_profile, catalog=catalog, config=cfg, params=p, runner=runner)


def click_image(rgb: np.ndarray, shape, key: str) -> tuple[int, int] | None:
    """Show ``rgb``; return a new click as (row, col) in image pixels."""
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates

        click = streamlit_image_coordinates(rgb, width=DISPLAY_WIDTH, key=key)
    except Exception:  # component unavailable (e.g. in tests)
        st.image(rgb, width=DISPLAY_WIDTH)
        return None
    if not click or click == st.session_state.get(f"{key}_last"):
        return None
    st.session_state[f"{key}_last"] = click
    h, w = shape
    return (int(round(click["y"] * h / click["height"])),
            int(round(click["x"] * w / click["width"])))  # fmt: skip


def step(n: int, title: str) -> None:
    st.markdown(f"### {n} · {title}")


# --- page ---------------------------------------------------------------------------------------
def main() -> None:
    """The page: six steps, top to bottom."""
    models = get_models()
    st.title("Virtual FFR and stent planning")
    st.caption(f"**{DISCLAIMER}** A research prototype, not a medical device: never use it to "
               "decide treatment for a real patient.")  # fmt: skip
    if models.segmentation is None:
        st.error("No segmentation model found (run `vstent export-seg`).")
        st.stop()

    physics = get_settings()[0]
    with st.sidebar:
        st.header("Settings")
        pa_mmhg = st.number_input("Aortic pressure (mmHg)", 50.0, 150.0,
                                  float(physics.pa_mmhg), 1.0,
                                  help="Mean aortic pressure during hyperaemia.")  # fmt: skip
        own_px = st.toggle("Enter the pixel size", key="own_px",
                           help="Use a calibrated pixel size (e.g. from the catheter) instead of "
                           "the file's.")  # fmt: skip
        px_value = st.number_input("mm per pixel", 0.05, 1.0, 0.30, 0.01, disabled=not own_px)
        st.divider()
        st.caption("The full research view (synthetic vessels, two views, vessel trees, every "
                   "interval): `vstent app --advanced`.")  # fmt: skip

    # 1 · load -------------------------------------------------------------------------------------
    step(1, "Load a cine angiogram")
    sources = ["Upload a cine run (DICOM)", guided.DEMOS["synthetic"]]
    if guided.CADICA_EXAMPLE.exists():
        sources.append(guided.DEMOS["cadica"])
    source = st.radio("Source", sources, horizontal=True, key="source",
                      label_visibility="collapsed")  # fmt: skip
    run = None
    if source == sources[0]:
        up = st.file_uploader("A whole run as exported by the X-ray system (multi-frame DICOM)",
                              type=["dcm"])  # fmt: skip
        if up is not None:
            data = up.getvalue()
            if not is_multiframe_dicom(data, up.name):
                st.error("This file has one frame. Upload the whole cine run (multi-frame DICOM).")
                st.stop()
            run = read_run_bytes(data, up.name)
    else:
        run = load_demo("synthetic" if source == guided.DEMOS["synthetic"] else "cadica")
    if run is None:
        st.info("Upload a cine run, or pick a demo above.")
        st.stop()
    rf = run_frames(run.sha256, run)
    st.success(f"**{run.name}**: " + " · ".join(guided.run_summary(run, rf)))

    # 2 · frame ------------------------------------------------------------------------------------
    step(2, "Choose the frame")
    st.write("Vessels are measured at **end-diastole**: the heart is still and the arteries are "
             "longest. These are the end-diastolic frames with the vessels full of contrast. "
             "★ marks the recommended one.")  # fmt: skip
    choices = guided.frame_choices(rf)
    rec = rf.analysis_frame
    cols = st.columns(max(len(choices), 3))
    for c, f in zip(cols, choices, strict=False):
        c.image(guided.thumbnail(run.frames[f]), caption=f"{'★ ' if f == rec else ''}Frame {f}",
                width="stretch")  # fmt: skip
    frame = st.radio("Frame", choices, index=choices.index(rec), horizontal=True,
                     format_func=lambda f: f"★ Frame {f}" if f == rec else f"Frame {f}",
                     key=f"frame_{run.sha256[:10]}", label_visibility="collapsed")  # fmt: skip
    with st.expander("Use any other frame"):
        if st.toggle("Pick a frame myself", key=f"anyframe_{run.sha256[:10]}"):
            frame = st.slider("Frame", 0, len(run) - 1, frame, key=f"slider_{run.sha256[:10]}")
            st.caption("Frames outside end-diastole make the heart look smaller and the vessels "
                       "shorter.")  # fmt: skip

    # 3 · vessel -----------------------------------------------------------------------------------
    step(3, "Vessel and lesions")
    image = run.frames[frame]
    frame_key = f"{run.sha256}:{frame}"
    prob = segment(frame_key, image)
    if st.session_state.get("path_key") != frame_key:
        try:
            a, b = guided.auto_path(prob, px_value if own_px else run.pixel_spacing_mm)
        except ValueError:
            st.error("No vessel was found on this frame. Choose another frame.")
            st.stop()
        st.session_state.update(path_key=frame_key, start=a, end=b, picking=None)
    mm_px = px_value if own_px else run.pixel_spacing_mm
    start, end = st.session_state.start, st.session_state.end
    analysis = None
    try:
        analysis = measure(f"{frame_key}:{start}:{end}:{mm_px}:{pa_mmhg}", prob, image, start, end,
                           mm_px, pa_mmhg)  # fmt: skip
    except ValueError as exc:
        st.warning(f"The path could not be measured: {exc}. Click new start and end points.")
    lesions = guided.lesion_marks(analysis.qca) if analysis is not None else []

    img_col, info_col = st.columns([3, 2])
    with info_col:
        picking = st.session_state.get("picking")
        if picking is None:
            st.write("The yellow line is the vessel path, found automatically: from the widest "
                     "end along the main vessel (at most 100 mm). Red circles mark narrowings of "
                     "50% or more. Not the vessel you want? Press **Change path**.")  # fmt: skip
            b1, b2, b3 = st.columns(3)
            if b1.button("Change path", type="primary", key="change"):
                st.session_state.picking = "start"
                st.rerun()
            if b2.button("Automatic path", key="auto"):
                st.session_state.path_key = None
                st.rerun()
            if b3.button("Reverse", key="reverse", help="Swap the start and the end."):
                st.session_state.update(start=end, end=start)
                st.rerun()
        else:
            where = ("proximal, near the catheter" if picking == "start" else
                     "distal, beyond the lesion")  # fmt: skip
            st.info(f"**Click the {picking} point** of the vessel on the image ({where}).")
            if st.button("Cancel", key="cancel"):
                st.session_state.picking = None
                st.rerun()
        if lesions:
            st.markdown("**Narrowings on this path**")
            for m in lesions:
                st.markdown(f"- {m.text()}")
        elif analysis is not None:
            st.write("No narrowing of 50% or more on this path.")
        if analysis is not None and guided.mild_count(analysis.qca):
            st.caption(f"{guided.mild_count(analysis.qca)} mild narrowing(s) of 30–49% not marked.")
        if analysis is not None:
            for w in analysis.warnings:
                st.caption(f"ⓘ {w}")
    with img_col:
        rgb = guided.overlay(image, prob, None if analysis is None else analysis.qca, lesions,
                             points=[start, end])  # fmt: skip
        pt = click_image(rgb, image.shape, f"click_{run.sha256[:10]}_{frame}")
        if pt is not None and st.session_state.get("picking"):
            if st.session_state.picking == "start":
                st.session_state.update(start=pt, picking="end")
            else:
                st.session_state.update(end=pt, picking=None)
            st.rerun()
        st.caption(f"Frame {frame}: blue = vessels found, yellow = path, green dot = start, orange "
                   "dot = end, red = narrowings.")  # fmt: skip
    if analysis is None:
        st.stop()

    # 4 · virtual FFR ------------------------------------------------------------------------------
    step(4, "Virtual FFR")
    cons = heartbeats(f"{frame_key}:{start}:{end}:{mm_px}:{pa_mmhg}", rf, frame, start, end, mm_px,
                      pa_mmhg, analysis)  # fmt: skip
    chosen = cons.chosen
    q = chosen.analysis.qca
    params = _params(pa_mmhg)
    pre = core.evaluate(q.profile, params, params.pa, params.pv)
    mc = uncertainty(f"{chosen.frame}:{chosen.start}:{chosen.end}:{mm_px}:{pa_mmhg}", q.profile,
                     pa_mmhg, q.mm_per_px, q.calibration_assumed)  # fmt: skip
    ucfg = get_uncertainty()[0]
    pre_a = core.assess(pre, mc.pre_samples, ucfg.pre_pci_threshold, ucfg.level)
    v = guided.verdict(cons.ffr, pre_a.flagged)
    stable, stable_text = guided.stability(cons)
    left, right = st.columns([2, 3])
    with left:
        st.metric("Virtual FFR at the end of the path", f"{cons.ffr:.2f}")
        st.markdown(f"#### :{v.color}[{v.label}]")
        st.write(v.sentence)
        if pre_a.inputs is not None:
            st.write(f"Likely range ({pre_a.inputs.level:.0%}): {pre_a.inputs.low:.2f} to "
                     f"{pre_a.inputs.high:.2f}.")  # fmt: skip
        (st.info if stable else st.warning)(stable_text)
        if q.calibration_assumed:
            st.caption("ⓘ No pixel size: 0.30 mm assumed. Diameters, lengths and FFR depend on it.")
    with right:
        st.plotly_chart(guided.ffr_chart(pre, None, guided.lesion_marks(q)), key="ffr_now")
    if chosen.frame != frame:
        st.caption(f"Frame {chosen.frame} is the heartbeat closest to the median, so the stent "
                   "step below uses it (the path followed the heart there).")  # fmt: skip

    # 5 · stent ------------------------------------------------------------------------------------
    step(5, "Stent")
    profile = q.profile
    vessel = hashlib.sha1(profile.s.tobytes() + profile.r.tobytes()).hexdigest()[:12]
    catalog = get_settings()[2]
    length_mm = m_to_mm(profile.length)
    if st.button("Suggest a stent", type="primary", key="suggest"):
        st.session_state[f"plans_{vessel}"] = suggest(f"{vessel}:{pa_mmhg}", profile, pa_mmhg)
    plans = st.session_state.get(f"plans_{vessel}")
    stent = None
    if plans is not None and plans.plans:
        labels = [guided.stent_text(list(p.stents), p.ffr) for p in plans.plans]
        pick = st.radio("Suggested plans (shortest first)", range(len(labels)),
                        format_func=lambda k: f"{'Best: ' if k == 0 else ''}{labels[k]}",
                        key=f"plan_{vessel}")  # fmt: skip
        chosen_plan = plans.plans[pick]
        if chosen_plan.meets_target:
            st.caption(plans.message)
        else:
            st.warning(f"No plan reaches the target FFR {plans.target:.2f}; the best reaches "
                       f"{chosen_plan.ffr:.2f}. Check that the path follows one vessel, or that "
                       "the narrowing is not elsewhere.")  # fmt: skip
        base = chosen_plan.stents[0]
    elif plans is not None:
        st.warning(plans.message)
        base = None
    else:
        base = None
        st.write("Press **Suggest a stent** for the shortest stent that brings the virtual FFR "
                 "back above the target, or place one yourself.")  # fmt: skip
    own = st.toggle("Place or adjust the stent myself", key=f"own_{vessel}", value=False)
    if own or base is not None:
        worst = max(guided.lesion_marks(q), key=lambda m: m.ds_pct, default=None)
        d0 = m_to_mm(base.diameter) if base else 3.0
        l0 = m_to_mm(base.length) if base else 18.0
        s0 = m_to_mm(base.s_start) if base else max(0.0, (worst.at_mm if worst else 10.0) - 9.0)
        lengths = [x for x in catalog.lengths_mm if x <= length_mm]  # stents that fit the path
        if own and not lengths:
            shortest = min(catalog.lengths_mm)
            st.info(f"The path is {length_mm:.0f} mm long: shorter than the shortest stent "
                    f"({shortest:g} mm). Choose a longer path in step 3.")  # fmt: skip
            stent = base
        elif own:
            c1, c2, c3 = st.columns(3)
            diameters = catalog.diameters_mm
            dia = c1.selectbox("Diameter (mm)", diameters,
                               index=int(np.argmin(np.abs(np.array(diameters) - d0))),
                               key=f"dia_{vessel}_{pick if base else 'own'}")  # fmt: skip
            ln = c2.selectbox("Length (mm)", lengths,
                              index=int(np.argmin(np.abs(np.array(lengths) - l0))),
                              key=f"len_{vessel}_{pick if base else 'own'}")  # fmt: skip
            max_start = max(0.0, float(np.floor((length_mm - ln) * 2) / 2))
            s_start = c3.slider("Start (mm along the path)", 0.0, max(max_start, 0.5),
                                float(min(max(s0, 0.0), max_start)), 0.5,
                                key=f"start_{vessel}_{pick if base else 'own'}")  # fmt: skip
            stent = Stent(mm_to_m(min(s_start, max_start)), mm_to_m(ln), mm_to_m(dia))
        else:
            stent = base
    post = stented = None
    if stent is not None:
        planner_cfg = get_settings()[1]
        stented, post = core.stent_update(profile, [stent], pre, params, params.pa, params.pv, None,
                                          edge=mm_to_m(planner_cfg.edge_mm))  # fmt: skip
        a_col, b_col = st.columns([3, 2])
        with b_col:
            delta = post.distal_solver - pre.distal_solver
            st.metric("Virtual FFR after the stent", f"{post.distal_solver:.2f}",
                      f"{delta:+.2f} (now {pre.distal_solver:.2f})")  # fmt: skip
            after = guided.verdict(post.distal_solver, False)
            st.markdown(f":{after.color}[{after.label} after stenting]")
            st.write(guided.stent_text([stent]))
            for w in core.stent_warnings(profile, [stent], planner_cfg.max_diameter_ratio):
                st.warning(w)
            st.plotly_chart(guided.ffr_chart(pre, post, guided.lesion_marks(q)), key="ffr_after")
        with a_col:
            img_c = run.frames[chosen.frame]
            st.image(guided.overlay(img_c, chosen.analysis.prob, q, guided.lesion_marks(q), [stent],
                                    [chosen.start, chosen.end]), width=DISPLAY_WIDTH)  # fmt: skip
            st.caption(f"Frame {chosen.frame}: the stent in orange.")
        with st.expander("Diameter along the path"):
            st.plotly_chart(core.profile_figure(profile, stented, [stent]), key="diameters")

    # 6 · report -----------------------------------------------------------------------------------
    step(6, "Report")
    state = core.CaseState(
        "image", params, params.pa, params.pv, profile, pre, [stent] if stent else [], post, q,
        run.frame(chosen.frame), [chosen.start, chosen.end], description=f"{run.name}, frame "
        f"{chosen.frame}", warnings=list(chosen.analysis.warnings), uncertainty=mc,
        pre_assessment=pre_a, consensus=cons.as_dict(),
    )  # fmt: skip
    if plans is not None:
        state.plans = plans
    r1, r2 = st.columns(2)
    r1.download_button("Download the report (JSON)", core.report_json(state, models),
                       file_name="vstent_report.json", mime="application/json")  # fmt: skip
    r2.download_button("Download the charts (PNG)", core.report_png(state),
                       file_name="vstent_report.png", mime="image/png")  # fmt: skip
    st.caption(DISCLAIMER)


if __name__ == "__main__":  # Streamlit runs the script; a worker process must not
    main()
