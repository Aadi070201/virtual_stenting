"""ED-frame labelling page (P4l; research use only). Launched by ``vstent label-ed``.

A reader scrolls through each run's frames and marks one end-diastolic frame per heart beat:
the frame where the heart and its arteries are largest, just before they contract. The logic
is in :mod:`vstent.app.ed_labels`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ when run by Streamlit

import streamlit as st  # noqa: E402

from vstent import DISCLAIMER  # noqa: E402
from vstent.app import ed_labels, guided  # noqa: E402


@st.cache_resource
def cases():
    return ed_labels.label_set()


@st.cache_resource(max_entries=4, show_spinner="Loading the run…")
def load(key: str, _case):
    return guided.load_case(_case)


def mark(key: str, frame: int) -> None:
    frames = set(st.session_state.get(key, ()))
    frames.symmetric_difference_update({frame})  # a second press removes the mark
    st.session_state[key] = tuple(sorted(frames))


def step(slider: str, delta: int, n: int) -> None:
    st.session_state[slider] = min(max(st.session_state.get(slider, 0) + delta, 0), n - 1)


def main() -> None:
    st.set_page_config(page_title="vstent: ED labels", page_icon="🫀", layout="wide")
    st.caption(DISCLAIMER)
    st.markdown("## End-diastolic frames: expert labels")
    reader = st.sidebar.text_input("Your name (labels are saved per reader)", key="reader")
    if not reader.strip():
        st.info("Enter your name in the sidebar to start.")
        st.stop()
    all_cases = cases()
    done = {c.key for c in all_cases if ed_labels.load_label(reader, c.key)}
    st.sidebar.progress(len(done) / max(len(all_cases), 1),
                        text=f"{len(done)} of {len(all_cases)} runs labelled")  # fmt: skip
    pick = st.sidebar.selectbox("Run", range(len(all_cases)), key="run",
                                format_func=lambda k: ("✓ " if all_cases[k].key in done else "")
                                + all_cases[k].label)  # fmt: skip
    case = all_cases[pick]
    run = load(case.key, case)
    n = len(run.frames)
    marks_key, slider = f"marks_{case.key}", f"frame_{case.key}"
    if marks_key not in st.session_state:
        saved = ed_labels.load_label(reader, case.key)
        st.session_state[marks_key] = saved.frames if saved else ()
    st.markdown(
        "Mark **one frame per heart beat**: the frame where the heart and its arteries "
        "are largest, just before they contract (end-diastole). Press again to unmark."
    )
    left, right = st.columns([3, 1])
    with right:
        frame = st.slider("Frame", 0, n - 1, key=slider)
        b1, b2 = st.columns(2)
        b1.button("◀ Previous", on_click=step, args=(slider, -1, n), width="stretch")
        b2.button("Next ▶", on_click=step, args=(slider, 1, n), width="stretch")
        marked = frame in st.session_state[marks_key]
        st.button("Unmark this frame" if marked else "Mark ED here", type="primary",
                  on_click=mark, args=(marks_key, frame), width="stretch",
                  key="mark")  # fmt: skip
        frames = st.session_state[marks_key]
        st.markdown(f"**ED frames:** {', '.join(map(str, frames)) or 'none yet'}")
        if run.frame_time_ms:
            st.caption(f"{1000 / run.frame_time_ms:.1f} frames per second · {n} frames")
        if st.button("Save", key="save", disabled=not frames, width="stretch"):
            ed_labels.save_label(ed_labels.EdLabel(case.key, reader, frames, n))
            st.success("Saved.")
    with left:
        st.image(run.frames[frame], width=640,
                 caption=f"Frame {frame}{' · ED' if frame in frames else ''}")  # fmt: skip


if __name__ == "__main__":
    main()
