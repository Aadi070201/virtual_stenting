"""P4k-J: AngioCAD labels (read from the .xlsx with the standard library) and the side summary."""

import zipfile

import pytest

from vstent.data import angiocad
from vstent.validation import angiocad_check as ac

HEADER = ["ID", "Right Coronary Series", "Left Coronary Series", *angiocad.SIDES["left"],
          *angiocad.SIDES["right"]]  # fmt: skip


def _col(k: int) -> str:
    return chr(ord("A") + k) if k < 26 else "A" + chr(ord("A") + k - 26)


def _workbook(path, table):
    strings, rows = [], []
    for r, values in enumerate(table, 1):
        cells = []
        for k, v in enumerate(values):
            ref = f"{_col(k)}{r}"
            if v is None:
                continue
            if isinstance(v, int):
                cells.append(f'<c r="{ref}"><v>{v}</v></c>')
            else:
                strings.append(v)
                cells.append(f'<c r="{ref}" t="s"><v>{len(strings) - 1}</v></c>')
        rows.append(f'<row r="{r}">{"".join(cells)}</row>')
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    sheet = f"<worksheet {ns}><sheetData>{''.join(rows)}</sheetData></worksheet>"
    sst = f"<sst {ns}>" + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/sharedStrings.xml", sst)


def test_labels_from_xlsx(tmp_path):
    left_ok = ["NL"] * len(angiocad.SIDES["left"])
    right_bad = ["NL", "76-90", "NL", "1-25", "NL"]
    left_mild = ["1-25"] + ["NL"] * (len(angiocad.SIDES["left"]) - 1)
    _workbook(tmp_path / angiocad.LABELS_FILE, [
        HEADER,
        [1, "1-2", "3-5", *left_ok, *right_bad],
        [2, "4", "1-3, 6", *left_mild, *["NL"] * 5],
        [None, None, None],
    ])  # fmt: skip
    labels = angiocad.load_labels(tmp_path)
    assert [p.patient for p in labels] == ["1", "2"]
    a, b = labels
    assert a.series == {"right": (1, 2), "left": (3, 4, 5)} and b.series["left"] == (1, 2, 3, 6)
    assert a.status("right") == "significant" and a.worst("right") == "76-90"
    assert a.status("left") == "normal" and b.status("left") == "mild"
    assert angiocad.parse_series(None) == () and angiocad.parse_series("7") == (7,)
    assert ac.tasks(labels)[:2] == [("1", "right", 1), ("1", "right", 2)]


def test_side_summary():
    grades = {s: "NL" for segs in angiocad.SIDES.values() for s in segs}
    sig = angiocad.PatientLabels("1", {"right": (1, 2), "left": (3,)},
                                 {**grades, "Mid RCA": "91-99"})  # fmt: skip
    nor = angiocad.PatientLabels("2", {"right": (1,), "left": (2,)}, grades)
    rows = [
        {"patient": "1", "side": "right", "series": 1, "ffr": 0.86, "marked": 0, "flagged": False},
        {"patient": "1", "side": "right", "series": 2, "ffr": 0.74, "marked": 1, "flagged": True},
        {"patient": "1", "side": "left", "series": 3, "ffr": 0.95, "marked": 0, "flagged": False},
        {"patient": "2", "side": "right", "series": 1, "ffr": 0.79, "marked": 0, "flagged": True},
        {"patient": "2", "side": "left", "series": 2, "error": "missing: no frame folder"},
    ]  # fmt: skip
    s = ac.summarise([sig, nor], rows)
    assert s["significant"]["flagged"] == {"n": 1, "k": 1, "share": 1.0}
    assert s["normal"]["flagged"]["n"] == 2 and s["normal"]["flagged"]["k"] == 1  # 1 left of 1
    assert s["sides_measured"] == 3 and s["series_missing"] == 1
    assert s["by_worst"]["91-99"]["k"] == 1
    assert s["normal_series"]["n"] == 2 and s["normal_series"]["ffr_le_080"] == pytest.approx(0.5)
    assert s["errors"] == {"missing": 1}
