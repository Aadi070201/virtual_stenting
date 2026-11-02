"""P4l: expert ED labels: storage, matching, agreement, and the labelling page."""

import pytest
from streamlit.testing.v1 import AppTest

from vstent.app import ed_labels as el
from vstent.cli import LABEL_SCRIPT
from vstent.data import cadica


def test_storage_matching_and_agreement(tmp_path):
    el.save_label(el.EdLabel("cadica:p1_v2", "Dr A", (3, 13, 24), 40), tmp_path)
    el.save_label(el.EdLabel("cadica:p1_v2", "Dr B", (4, 15, 24, 35), 40), tmp_path)
    assert el.load_label("Dr A", "cadica:p1_v2", tmp_path).frames == (3, 13, 24)
    assert el.match((3, 13, 24), (4, 15, 24, 35)) == 2  # 3~4 and 24~24; 13 vs 15 is 2 apart
    a = el.agreement("Dr A", "Dr B", tmp_path)
    assert a["runs"] == 1 and a["agree"] == 2 and a["share"] == pytest.approx(0.5)


def test_labelling_page(monkeypatch, tmp_path):
    monkeypatch.setattr(cadica, "RAW_DIR", tmp_path / "no_cadica")  # synthetic cases only
    monkeypatch.setattr(el, "LABEL_DIR", tmp_path / "labels")
    import streamlit as st

    st.cache_resource.clear()
    at = AppTest.from_file(str(LABEL_SCRIPT), default_timeout=300).run()
    assert not at.exception and "Enter your name" in at.info[0].value
    at.text_input(key="reader").set_value("Dr A").run()
    at.button(key="mark").click().run()
    assert not at.exception, [e.value for e in at.exception]
    at.button(key="save").click().run()
    assert not at.exception and any(p.suffix == ".json" for p in (tmp_path / "labels").rglob("*"))
