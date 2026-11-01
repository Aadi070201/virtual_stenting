import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from vstent.cli import app
from vstent.regulatory.build import REG_DIR, write_all
from vstent.regulatory.soup import (
    Lock,
    check_soup,
    environment,
    load_soup,
    release_packages,
    third_party_imports,
)
from vstent.regulatory.trace import RiskPolicy, check_trace, load_trace

TOY_LOCK = """\
version = 1
revision = 3
requires-python = ">=3.11"

[[package]]
name = "demo"
version = "0.1.0"
source = { editable = "." }
dependencies = [
    { name = "alpha" },
    { name = "gamma", version = "1.0", source = { registry = "https://pypi.org/simple" }, \
marker = "python_full_version < '3.12'" },
    { name = "gamma", version = "2.0", source = { registry = "https://pypi.org/simple" }, \
marker = "python_full_version >= '3.12'" },
]

[package.optional-dependencies]
app = [{ name = "beta" }]
dev = [{ name = "tool" }]

[[package]]
name = "alpha"
version = "1.2"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "winonly", marker = "sys_platform == 'win32'" },
    { name = "delta", extra = ["fast"] },
]

[[package]]
name = "beta"
version = "3.0+cu1"
source = { registry = "https://download.example.org/whl" }

[[package]]
name = "delta"
version = "0.5"
source = { registry = "https://pypi.org/simple" }

[package.optional-dependencies]
fast = [{ name = "epsilon" }]

[[package]]
name = "epsilon"
version = "1.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "gamma"
version = "1.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "gamma"
version = "2.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "tool"
version = "9.9"
source = { registry = "https://pypi.org/simple" }
dependencies = [{ name = "alpha" }]

[[package]]
name = "winonly"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
"""

POLICY = {
    "use": "one analysed vessel",
    "severity": {i: {"name": f"S{i}", "meaning": "m"} for i in range(1, 6)},
    "probability": {i: {"name": f"P{i}", "meaning": "m"} for i in range(1, 6)},
    "acceptable_max": 4,
    "unacceptable_min": 10,
}


def _requirement(rid, **kw):
    return {"id": rid, "title": "t", "text": "x", "kind": "physics",
            "code": ["src/vstent/core.py::solve"], **kw}  # fmt: skip


def _hazard(hid, controls):
    return {"id": hid, "title": "t", "cause": "c", "situation": "s", "harm": "h",
            "severity": 4, "probability": 5, "basis": "b", "controls": controls,
            "residual_probability": 2, "residual_basis": "rb", "needed": "n"}  # fmt: skip


def _toy_repo(root: Path) -> None:
    """A minimal repository: one module importing alpha, one test, one report."""
    (root / "src/vstent").mkdir(parents=True)
    (root / "src/vstent/core.py").write_text(
        "import os\nimport alpha.sub\n\n\ndef solve():\n    return 1\n", encoding="utf-8"
    )
    (root / "tests").mkdir()
    (root / "tests/test_core.py").write_text("def test_solve():\n    pass\n", encoding="utf-8")
    (root / "reports").mkdir()
    (root / "reports/r.md").write_text("Research use only.\n", encoding="utf-8")
    (root / "uv.lock").write_text(TOY_LOCK, encoding="utf-8")
    (root / ".python-version").write_text("3.14\n", encoding="utf-8")
    reg = root / REG_DIR
    reg.mkdir(parents=True)
    reqs = [_requirement("SR-01", tests=["tests/test_core.py::test_solve"],
                         reports=["reports/r.md"]),
            _requirement("SR-02", gap="not built yet")]  # fmt: skip
    (reg / "requirements.yaml").write_text(yaml.safe_dump({"requirements": reqs}), "utf-8")
    hz = {"policy": POLICY, "hazards": [_hazard("H-01", ["SR-01", "SR-02"])]}
    (reg / "hazards.yaml").write_text(yaml.safe_dump(hz), "utf-8")
    soup = {
        "soup": [
            {"name": n, "imports": imp, "purpose": "p", "relied_on": "r", "anomalies": "a"}
            for n, imp in (("alpha", ["alpha"]), ("beta", []), ("gamma", []), ("tool", []))
        ],
        "platform": [{"name": "CPython", "version": "3.14", "purpose": "p"}],
    }
    (reg / "soup.yaml").write_text(yaml.safe_dump(soup), "utf-8")


def test_lock_closure_follows_markers_extras_and_versions():
    lock = Lock(TOY_LOCK)
    linux = lock.closure(environment("linux", "3.14"), ("app",))
    assert sorted((n, v) for n, v, _ in linux) == [
        ("alpha", "1.2"), ("beta", "3.0+cu1"), ("delta", "0.5"), ("epsilon", "1.0"),
        ("gamma", "2.0"),
    ]  # fmt: skip
    win = lock.closure(environment("win32", "3.11"), ())
    assert {n: v for n, v, _ in win}["gamma"] == "1.0" and ("winonly", "1.0") in {
        (n, v) for n, v, _ in win
    }
    pkgs = {p.name: p for p in release_packages(lock, "3.14")}
    assert pkgs["winonly"].platforms == ("win32",) and pkgs["alpha"].platforms == (
        "linux",
        "win32",
    )
    assert pkgs["alpha"].direct and not pkgs["delta"].direct and pkgs["alpha"].scope == "runtime"
    assert pkgs["tool"].scope == "development" and pkgs["tool"].direct
    assert "gamma" in pkgs and pkgs["gamma"].version == "2.0"


def test_risk_policy_levels():
    pol = RiskPolicy.model_validate(POLICY)
    assert pol.level(4, 1) == "acceptable" and pol.level(5, 1) == "reduce"
    assert pol.level(3, 3) == "reduce" and pol.level(5, 2) == "unacceptable"
    with pytest.raises(ValueError, match="1–5"):
        RiskPolicy.model_validate({**POLICY, "severity": {1: {"name": "a", "meaning": "b"}}})


def test_trace_check_finds_every_broken_reference(tmp_path):
    _toy_repo(tmp_path)
    reg = tmp_path / REG_DIR
    reqs = [
        _requirement("SR-01", tests=["tests/test_core.py::test_gone", "test_core.py::x"]),
        _requirement("SR-01", code=["src/vstent/core.py::Missing", "src/none.py"],
                     reports=["reports/none.md"]),
    ]  # fmt: skip
    (reg / "requirements.yaml").write_text(yaml.safe_dump({"requirements": reqs}), "utf-8")
    hz = {"policy": POLICY, "hazards": [_hazard("H-01", ["SR-09"])]}
    (reg / "hazards.yaml").write_text(yaml.safe_dump(hz), "utf-8")
    with pytest.raises(ValueError) as err:
        load_trace(tmp_path)
    msg = str(err.value)
    expected = [
        "SR-01 is defined twice",
        "test_gone is not defined",
        "expected tests/",
        "Missing is not defined",
        "src/none.py does not exist",
        "reports/none.md does not exist",
        "control SR-09 is not a requirement",
    ]
    assert [part for part in expected if part not in msg] == []
    unverified = {"requirements": [_requirement("SR-03")]}
    (reg / "requirements.yaml").write_text(yaml.safe_dump(unverified), "utf-8")
    with pytest.raises(ValueError, match="no verification and no gap"):
        load_trace(tmp_path)


def test_generator_on_a_small_repository(tmp_path, monkeypatch):
    _toy_repo(tmp_path)
    written = write_all(tmp_path)
    assert sorted(p.name for p in written) == [
        "HAZARD_ANALYSIS.md", "REQUIREMENTS.md", "SOUP.md", "TRACEABILITY.md", "sbom.cdx.json",
    ]  # fmt: skip
    assert write_all(tmp_path, check=True) == []
    reg = tmp_path / REG_DIR
    hazards = (reg / "HAZARD_ANALYSIS.md").read_text("utf-8")
    assert "Research use only" in hazards and "| H-01 | t | 4 | 5 | 20 unacceptable |" in hazards
    assert "- **Controls:** SR-01 t; SR-02 t" in hazards
    trace = (reg / "TRACEABILITY.md").read_text("utf-8")
    assert "**SR-02 t:** not built yet" in trace and "| H-01 | SR-02 | t | gap | gap |" in trace
    soup = (reg / "SOUP.md").read_text("utf-8")
    assert "**6 packages are shipped** (3 declared directly" in soup
    assert "| alpha | 1.2 | core |" in soup and "| tool | 9.9 | not imported |" in soup
    bom = json.loads((reg / "sbom.cdx.json").read_text("utf-8"))
    purls = {c["name"]: c["purl"] for c in bom["components"]}
    assert (
        purls["beta"] == "pkg:pypi/beta@3.0%2Bcu1?repository_url=https://download.example.org/whl"
    )
    assert {c["name"]: c["scope"] for c in bom["components"]}["tool"] == "excluded"
    # a stale document is reported by --check, not rewritten
    (reg / "SOUP.md").write_text("edited by hand\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["regulatory", "--check"])
    assert result.exit_code == 1 and "Out of date: docs/regulatory/SOUP.md" in result.output
    assert (reg / "SOUP.md").read_text("utf-8") == "edited by hand\n"
    result = CliRunner().invoke(app, ["regulatory"])
    assert result.exit_code == 0 and "Wrote docs/regulatory/SOUP.md" in result.output
    # an import without a SOUP entry fails the build
    (tmp_path / "src/vstent/extra.py").write_text("import zeta\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["regulatory", "--check"])
    assert result.exit_code == 1 and "zeta is imported (src/vstent/extra.py)" in result.output


def test_every_imported_module_is_declared_soup():
    root = Path(".")
    lock = Lock.read(root)
    imports = third_party_imports(root)
    assert "numpy" in imports and "os" not in imports and "vstent" not in imports
    assert check_soup(load_soup(root), lock, imports) == []
    problems = check_soup(load_soup(root), lock, {**imports, "zeta": {"src/vstent/x.py"}})
    assert problems == [
        "zeta is imported (src/vstent/x.py) but belongs to no SOUP entry: declare it in "
        "pyproject.toml and soup.yaml"
    ]


def test_committed_documents_are_up_to_date():
    """docs/regulatory/ matches its sources, and every reference in them resolves."""
    assert check_trace(load_trace(Path(".")), Path(".")) == []
    assert write_all(Path("."), check=True) == [], "run `uv run vstent regulatory`"
