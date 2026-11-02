"""vstent guided app: one cine run, step by step (research use only).

Launched by ``vstent app``; the full research view (synthetic vessels, single images, two
views, trees, every interval) is ``vstent app --advanced``. The steps and their logic are in
:mod:`vstent.app.guided`.
"""

import hashlib
import os
import random

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

IMAGE_WIDTH = 620
UPLOAD = "Upload a DICOM cine run"


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


@st.cache_resource
def library() -> list[guided.Case]:
    return guided.case_library()


@st.cache_resource(max_entries=3, show_spinner="Loading the case…")
def load(key: str, _case: guided.Case):
    return guided.load_case(_case)


@st.cache_resource(max_entries=3, show_spinner="Reading every frame (about a minute)…")
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


@st.cache_resource(max_entries=16, show_spinner="Measuring the other heartbeats…")
def heartbeats(key: str, _rf, frame: int, start, end, mm_per_px, pa_mmhg, _reference):
    params = _params(pa_mmhg)

    def one(p, im, a, b):
        return analyse_vessel(p, a, b, params=params, mm_per_px=mm_per_px, image=im)

    return run_consensus(_rf, frame, start, end, get_models().segmentation.predict, one,
                         reference_analysis=_reference)  # fmt: skip


@st.cache_resource(max_entries=16, show_spinner="Estimating the range…")
def uncertainty(key: str, _profile, pa_mmhg: float, mm_per_px: float, assumed: bool):
    p = _params(pa_mmhg)
    return core.input_uncertainty(_profile, p, p.pa, p.pv, Measurement(mm_per_px, assumed),
                                  get_uncertainty())  # fmt: skip


@st.cache_resource(max_entries=16, show_spinner="Trying stent sizes and positions…")
def suggest(key: str, _profile, pa_mmhg: float):
    p, cfg, catalog = _params(pa_mmhg), get_settings()[1], get_settings()[2]
    runner = get_models().surrogate
    if runner is None:
        cfg = cfg.model_copy(update={"backend": "solver"})
    return plan_stents(_profile, catalog=catalog, config=cfg, params=p, runner=runner)


# --- pieces ---------------------------------------------------------------------------------------
def click_image(rgb: np.ndarray, shape, key: str) -> tuple[int, int] | None:
    """Show ``rgb``; return a new click as (row, col) in image pixels."""
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates

        click = streamlit_image_coordinates(rgb, width=IMAGE_WIDTH, key=key)
    except Exception:  # component unavailable (e.g. in tests)
        st.image(rgb, width=IMAGE_WIDTH)
        return None
    if not click or click == st.session_state.get(f"{key}_last"):
        return None
    st.session_state[f"{key}_last"] = click
    h, w = shape
    return (int(round(click["y"] * h / click["height"])),
            int(round(click["x"] * w / click["width"])))  # fmt: skip


def big_number(value: float, verdict: guided.Verdict, label: str) -> None:
    color = guided.HEX[verdict.color]
    st.markdown(f"<div style='font-size:0.9rem;color:#52514e'>{label}</div>"
                f"<div style='font-size:3.2rem;font-weight:700;color:{color};line-height:1.1'>"
                f"{value:.2f}</div><div style='color:{color};font-weight:600'>{verdict.label} · "
                f"{verdict.sentence}</div>", unsafe_allow_html=True)  # fmt: skip


def ffr_guide() -> None:
    rows = "".join(f"<div><span style='color:{guided.HEX[c]}'>●</span> <b>{rng}</b>: {text}</div>"
                   for c, rng, text in guided.FFR_GUIDE)  # fmt: skip
    st.markdown(f"<div style='font-size:0.85rem;line-height:1.6'>{rows}</div>",
                unsafe_allow_html=True)  # fmt: skip
    st.caption("From pressure-wire FFR studies (FAME; ESC 2019). Research use only.")


@st.fragment
def frame_picker(run, rf, sha: str, expert: frozenset) -> None:
    """Scroll the run (fast: nothing is measured); "Analyse" commits the frame. ``expert``: the
    frames with an expert lesion box (CADICA)."""
    key = f"frame_{sha}"
    current = st.session_state.get(key, rf.analysis_frame)
    every = st.toggle("All frames", key=f"every_{sha}",
                      help="Every frame (★ best, ◆ expert box).")  # fmt: skip
    eds = sorted(set(rf.ed.frames)) if rf.ed.found else []
    options = (list(range(len(run))) if every or not eds else
               sorted({*eds, *expert, rf.analysis_frame, current}))  # fmt: skip
    f = st.select_slider("Frame", options, value=current if current in options else
                         rf.analysis_frame, key=f"scroll_{sha}_{every}",
                         format_func=lambda x: guided.frame_label(x, rf, every, expert),
                         label_visibility="collapsed")  # fmt: skip
    left, right = st.columns([4, 1])
    left.image(run.frames[f], width=IMAGE_WIDTH)
    with right:
        st.markdown(f"**Frame {f}**" + (" ★" if f == rf.analysis_frame else ""))
        if f == current:
            st.success("Analysed below")
        elif st.button("Analyse this frame", type="primary", key=f"use_{sha}_{f}"):
            st.session_state[key] = f
            st.rerun()


def reset_study() -> None:
    for k in list(st.session_state):
        if k != "case":
            del st.session_state[k]


# --- page ---------------------------------------------------------------------------------------
def main() -> None:
    """The page: case, frame, vessel, virtual FFR, stent, report."""
    st.set_page_config(page_title="vstent: virtual FFR", page_icon="🫀", layout="wide")
    models = get_models()
    st.markdown("## Virtual FFR & stent planning")
    st.caption(f"{DISCLAIMER} Not a medical device.")
    if models.segmentation is None:
        st.error("No segmentation model found (run `vstent export-seg`).")
        st.stop()
    cases = library()
    labels = [UPLOAD] + [c.label for c in cases]
    physics = get_settings()[0]
    with st.sidebar:
        st.markdown("### Study")
        b1, b2 = st.columns(2)
        if b1.button("🎲 Random", help="A random case from the library", width="stretch"):
            reset_study()
            st.session_state.case = random.choice(labels[1:])
        if b2.button("New study", help="Clear everything and pick a case", width="stretch"):
            reset_study()
            st.session_state.case = labels[1]
        label = st.selectbox("Case", labels, index=1, key="case",
                             help="Synthetic beating hearts, then real CADICA runs (most severe "
                             "lesions first). Type to search.")  # fmt: skip
        with st.expander("Settings"):
            pa_mmhg = st.number_input("Aortic pressure (mmHg)", 50.0, 150.0,
                                      float(physics.pa_mmhg), 1.0)  # fmt: skip
            own_px = st.toggle("Set pixel size", key="own_px",
                               help="E.g. from catheter calibration; otherwise the file's, or "
                               "0.30 mm assumed.")  # fmt: skip
            px_value = st.number_input("mm per pixel", 0.05, 1.0, 0.30, 0.01, disabled=not own_px)
        st.markdown("### FFR guide")
        ffr_guide()

    # case -------------------------------------------------------------------------------------
    case = None if label == UPLOAD else next(c for c in cases if c.label == label)
    if case is None:
        up = st.file_uploader("Cine run (multi-frame DICOM)", type=["dcm"])
        if up is None:
            st.stop()
        data = up.getvalue()
        if not is_multiframe_dicom(data, up.name):
            st.error("One frame only: upload the whole cine run.")
            st.stop()
        run = read_run_bytes(data, up.name)
    else:
        run = load(case.key, case)
    rf = run_frames(run.sha256, run)
    sha = run.sha256[:12]
    st.caption(" · ".join([run.name, *guided.run_summary(run, rf)]))

    # 1 · frame --------------------------------------------------------------------------------
    st.markdown("#### 1 · Frame")
    expert = frozenset(b[0] for b in (case.boxes if case else ()))
    frame_picker(run, rf, sha, expert)
    frame = st.session_state.get(f"frame_{sha}", rf.analysis_frame)

    # 2 · vessel and lesions -------------------------------------------------------------------
    st.markdown("#### 2 · Vessel and lesions")
    image = run.frames[frame]
    frame_key = f"{run.sha256}:{frame}"
    prob = segment(frame_key, image)
    mm_px = px_value if own_px else run.pixel_spacing_mm
    if st.session_state.get("path_key") != frame_key:
        try:
            a, b = guided.auto_path(prob, mm_px)
        except ValueError:
            st.error("No vessel found on this frame: pick another one.")
            st.stop()
        st.session_state.update(path_key=frame_key, start=a, end=b, picking=None)
    start, end = st.session_state.start, st.session_state.end
    analysis = None
    try:
        analysis = measure(f"{frame_key}:{start}:{end}:{mm_px}:{pa_mmhg}", prob, image, start,
                           end, mm_px, pa_mmhg)  # fmt: skip
    except ValueError:
        st.warning("This path cannot be measured: press Change path and click two points on "
                   "the vessel.")  # fmt: skip
    q0 = None if analysis is None else analysis.qca
    lesions = guided.lesion_marks(q0) if q0 is not None else []
    boxes = [b[1:] for b in (case.boxes if case else ()) if b[0] == frame]
    img_col, side = st.columns([4, 2])
    with side:
        picking = st.session_state.get("picking")
        if picking:
            st.info(f"Click the **{picking}** point on the image.")
            if st.button("Cancel", key="cancel"):
                st.session_state.picking = None
                st.rerun()
        else:
            c1, c2, c3 = st.columns(3)
            if c1.button("Change path", type="primary", key="change",
                         help="Click the start, then the end, on the image."):  # fmt: skip
                st.session_state.picking = "start"
                st.rerun()
            if c2.button("Auto", key="auto", help="The automatic path again."):
                st.session_state.path_key = None
                st.rerun()
            if c3.button("⇄", key="reverse", help="Swap start and end."):
                st.session_state.update(start=end, end=start)
                st.rerun()
            if boxes and st.button("Through the expert box", key="expert_path",
                                   help="A path through the most severe expert box."):  # fmt: skip
                ranked = sorted(boxes, key=lambda b: -guided.severity(b[4]))
                ends = next((e for b in ranked if (e := guided.box_path(prob, b[:4]))), None)
                if ends is None:
                    st.warning("No segmented vessel crosses the expert boxes.")
                else:
                    st.session_state.update(start=ends[0], end=ends[1])
                    st.rerun()
        show = st.toggle("Overlays", value=True, key="overlays",
                         help="Off: the plain angiogram.")  # fmt: skip
        for m in lesions:
            st.markdown(f"**{m.number}** · {m.ds_pct:.0f}% · {m.length_mm:.0f} mm · MLD "
                        f"{m.mld_mm:.2f} mm")  # fmt: skip
        if q0 is not None and not lesions:
            st.markdown("No narrowing ≥ 50%")
        if boxes:
            st.caption("Cyan: the experts' lesion box on this frame.")
        if analysis is not None and analysis.warnings:
            with st.expander("Notes"):
                for w in analysis.warnings:
                    st.caption(w)
    with img_col:
        rgb = (guided.overlay(image, prob, q0, lesions, points=[start, end], boxes=boxes) if show
               else guided.overlay(image, None))  # fmt: skip
        pt = click_image(rgb, image.shape, f"click_{sha}_{frame}")
        if pt is not None and st.session_state.get("picking"):
            if st.session_state.picking == "start":
                st.session_state.update(start=pt, picking="end")
            else:
                st.session_state.update(end=pt, picking=None)
            st.rerun()
    if lesions:
        zooms = st.columns(min(len(lesions), 4))
        for col, m in zip(zooms, lesions[:4], strict=False):
            col.image(guided.lesion_zoom(image, q0, m, edges=show), width="stretch",
                      caption=f"Lesion {m.number}: {m.ds_pct:.0f}% · yellow = edges")  # fmt: skip
    if analysis is None:
        st.stop()

    # 3 · virtual FFR --------------------------------------------------------------------------
    st.markdown("#### 3 · Virtual FFR")
    cons = heartbeats(f"{frame_key}:{start}:{end}:{mm_px}:{pa_mmhg}", rf, frame, start, end,
                      mm_px, pa_mmhg, analysis)  # fmt: skip
    chosen = cons.chosen
    q = chosen.analysis.qca
    params = _params(pa_mmhg)
    pre = core.evaluate(q.profile, params, params.pa, params.pv)
    mc = uncertainty(f"{chosen.frame}:{chosen.start}:{chosen.end}:{mm_px}:{pa_mmhg}", q.profile,
                     pa_mmhg, q.mm_per_px, q.calibration_assumed)  # fmt: skip
    ucfg = get_uncertainty()[0]
    pre_a = core.assess(pre, mc.pre_samples, ucfg.pre_pci_threshold, ucfg.level)
    stable, stable_text = guided.stability(cons)
    left, right = st.columns([2, 3])
    with left:
        big_number(cons.ffr, guided.verdict(cons.ffr), "Virtual FFR (end of the path)")
        if pre_a.inputs is not None:
            st.caption(f"Likely range {pre_a.inputs.low:.2f}–{pre_a.inputs.high:.2f}")
        n = len(cons.used)
        st.caption(f"{'✓' if stable else '⚠'} {n} heartbeat{'s' if n > 1 else ''}",
                   help=stable_text)  # fmt: skip
        if not stable:
            st.warning("Heartbeats disagree about 0.80: check the frame and the path.")
        if q.calibration_assumed:
            st.caption("Pixel size assumed (0.30 mm)", help="Set it under Settings.")
    with right:
        st.plotly_chart(guided.ffr_chart(pre, None, guided.lesion_marks(q)), key="ffr_now")

    # 4 · stent --------------------------------------------------------------------------------
    st.markdown("#### 4 · Stent")
    profile = q.profile
    vessel = hashlib.sha1(profile.s.tobytes() + profile.r.tobytes()).hexdigest()[:12]
    catalog, planner_cfg = get_settings()[2], get_settings()[1]
    length_mm = m_to_mm(profile.length)
    if guided.verdict(cons.ffr).color == "green":
        st.success("Above 0.80: no stent needed. You can still try one.")
    c1, c2 = st.columns([1, 3])
    if c1.button("Suggest a stent", type="primary", key="suggest"):
        st.session_state[f"plans_{vessel}"] = suggest(f"{vessel}:{pa_mmhg}", profile, pa_mmhg)
    own = c2.toggle("Adjust it myself", key=f"own_{vessel}")
    plans = st.session_state.get(f"plans_{vessel}")
    base = pick = None
    if plans is not None and plans.plans:
        pick = st.radio("Plans", range(len(plans.plans)), key=f"plan_{vessel}",
                        format_func=lambda k: guided.stent_text(list(plans.plans[k].stents),
                                                                plans.plans[k].ffr),
                        label_visibility="collapsed")  # fmt: skip
        base = plans.plans[pick].stents[0]
        if not plans.plans[pick].meets_target:
            st.warning(f"No plan reaches {plans.target:.2f}: check the path.")
    elif plans is not None:
        st.warning(plans.message)
    stent = base
    lengths = [x for x in catalog.lengths_mm if x <= length_mm]  # stents that fit the path
    if own and not lengths:
        st.info(f"Path {length_mm:.0f} mm: shorter than any stent.")
    elif own:
        worst = max(guided.lesion_marks(q), key=lambda m: m.ds_pct, default=None)
        d0 = m_to_mm(base.diameter) if base else 3.0
        l0 = m_to_mm(base.length) if base else 18.0
        s0 = m_to_mm(base.s_start) if base else max(0.0, (worst.at_mm if worst else 10.0) - 9.0)
        tag = f"{vessel}_{pick}"
        k1, k2, k3 = st.columns(3)
        dias = catalog.diameters_mm
        dia = k1.selectbox("Diameter (mm)", dias, key=f"dia_{tag}",
                           index=int(np.argmin(np.abs(np.array(dias) - d0))))  # fmt: skip
        ln = k2.selectbox("Length (mm)", lengths, key=f"len_{tag}",
                          index=int(np.argmin(np.abs(np.array(lengths) - l0))))  # fmt: skip
        max_start = max(0.0, float(np.floor((length_mm - ln) * 2) / 2))
        s_start = k3.slider("Start (mm)", 0.0, max(max_start, 0.5),
                            float(min(max(s0, 0.0), max_start)), 0.5,
                            key=f"start_{tag}")  # fmt: skip
        stent = Stent(mm_to_m(min(s_start, max_start)), mm_to_m(ln), mm_to_m(dia))
    post = None
    if stent is not None:
        stented, post = core.stent_update(profile, [stent], pre, params, params.pa, params.pv,
                                          None, edge=mm_to_m(planner_cfg.edge_mm))  # fmt: skip
        a_col, b_col = st.columns([3, 2])
        with a_col:
            rgb = guided.overlay(
                run.frames[chosen.frame],
                chosen.analysis.prob,
                q,
                guided.lesion_marks(q),
                [stent],
                [chosen.start, chosen.end],
            )
            st.image(rgb, width=IMAGE_WIDTH, caption=f"Frame {chosen.frame} · stent in orange")
        with b_col:
            after = post.distal_solver
            v_after = (guided.Verdict("Good result", "blue", "after the stent")
                       if after >= guided.POST_TARGET else guided.verdict(after))  # fmt: skip
            big_number(after, v_after, f"After the stent (now {pre.distal_solver:.2f})")
            st.caption(guided.stent_text([stent]))
            for w in core.stent_warnings(profile, [stent], planner_cfg.max_diameter_ratio):
                st.warning(w)
            st.plotly_chart(guided.ffr_chart(pre, post, guided.lesion_marks(q)),
                            key="ffr_after")  # fmt: skip
        with st.expander("Diameter along the path"):
            st.plotly_chart(core.profile_figure(profile, stented, [stent]), key="diameters")

    # 5 · report -------------------------------------------------------------------------------
    st.markdown("#### 5 · Report")
    state = core.CaseState(
        "image", params, params.pa, params.pv, profile, pre, [stent] if stent else [], post, q,
        run.frame(chosen.frame), [chosen.start, chosen.end],
        description=f"{run.name}, frame {chosen.frame}",
        warnings=list(chosen.analysis.warnings), uncertainty=mc, pre_assessment=pre_a,
        consensus=cons.as_dict(),
    )  # fmt: skip
    if plans is not None:
        state.plans = plans
    r1, r2, _ = st.columns([1, 1, 2])
    r1.download_button("Report (JSON)", core.report_json(state, models),
                       file_name="vstent_report.json", mime="application/json")  # fmt: skip
    r2.download_button("Charts (PNG)", core.report_png(state), file_name="vstent_report.png",
                       mime="image/png")  # fmt: skip


if __name__ == "__main__":  # Streamlit runs the script; a worker process must not
    main()
