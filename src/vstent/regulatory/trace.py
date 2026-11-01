"""Requirements, hazards and their traceability (IEC 62304 §5.2, §7; ISO 14971 §5–§7).

Sources: ``docs/regulatory/requirements.yaml`` and ``docs/regulatory/hazards.yaml`` (with the
risk policy: severity and probability scales and the acceptability thresholds). Every
reference is checked against the repository:

- code (``path`` or ``path::name``, ``name`` a top-level definition or ``Class.method``);
- tests (``tests/<file>.py::test_name``);
- reports (files under ``reports/``);
- each hazard's controls are requirements, each safety requirement controls a hazard, and a
  requirement without a verification names its gap.

A broken reference is an error, so a renamed test or function cannot silently leave the trace.
"""

import ast
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from pydantic import Field, model_validator

from vstent.config import StrictModel, load_yaml

REQUIREMENTS_FILE = Path("docs/regulatory/requirements.yaml")
HAZARDS_FILE = Path("docs/regulatory/hazards.yaml")

KINDS = {
    "labelling": "Labelling and information for safety",
    "input": "Inputs",
    "imaging": "Segmentation and QCA",
    "physics": "Physics",
    "surrogate": "Surrogate model",
    "uncertainty": "Uncertainty and grey zone",
    "planning": "Virtual stent and planner",
    "interface": "User interface and performance",
    "data": "Data protection, integrity and reproducibility",
}


class Requirement(StrictModel):
    """One software requirement (SRS item)."""

    id: str
    title: str
    text: str
    kind: str
    code: list[str] = Field(min_length=1)
    tests: list[str] = Field(default_factory=list)
    reports: list[str] = Field(default_factory=list)
    inspection: str | None = None
    gap: str | None = None

    @model_validator(mode="after")
    def _verified_or_gap(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: kind must be one of {', '.join(KINDS)}")
        if not (self.tests or self.reports or self.inspection or self.gap):
            raise ValueError(f"{self.id}: no verification and no gap stated")
        return self


class Level(StrictModel):
    name: str
    meaning: str


class RiskPolicy(StrictModel):
    """Severity and probability scales (1–5) and acceptability: risk index = S × P;
    ≤ ``acceptable_max`` acceptable, ≥ ``unacceptable_min`` unacceptable, between: reduce as
    far as possible and weigh against the benefit."""

    use: str  # what one "use" is, for the probability scale
    severity: dict[int, Level]
    probability: dict[int, Level]
    acceptable_max: int
    unacceptable_min: int

    @model_validator(mode="after")
    def _scales(self):
        for name, scale in (("severity", self.severity), ("probability", self.probability)):
            if sorted(scale) != [1, 2, 3, 4, 5]:
                raise ValueError(f"{name} scale must have levels 1–5")
        if not 1 <= self.acceptable_max < self.unacceptable_min <= 25:
            raise ValueError("need 1 ≤ acceptable_max < unacceptable_min ≤ 25")
        return self

    def level(self, severity: int, probability: int) -> str:
        """``"acceptable"``, ``"reduce"`` (ALARP + benefit–risk) or ``"unacceptable"``."""
        index = severity * probability
        if index <= self.acceptable_max:
            return "acceptable"
        return "unacceptable" if index >= self.unacceptable_min else "reduce"


class Hazard(StrictModel):
    """One hazard: the sequence of events from a cause to a harm, its risk before and after
    the risk controls (requirements), and the evidence still needed."""

    id: str
    title: str
    cause: str
    situation: str
    harm: str
    severity: int = Field(ge=1, le=5)
    probability: int = Field(ge=1, le=5)
    basis: str
    controls: list[str] = Field(min_length=1)
    residual_probability: int = Field(ge=1, le=5)
    residual_basis: str
    evidence: list[str] = Field(default_factory=list)  # reports with measured evidence
    needed: str


class HazardFile(StrictModel):
    policy: RiskPolicy
    hazards: list[Hazard]


class RequirementFile(StrictModel):
    requirements: list[Requirement]


@dataclass(frozen=True)
class Trace:
    """Validated requirements, hazards and policy."""

    requirements: list[Requirement]
    hazards: list[Hazard]
    policy: RiskPolicy

    def controls_of(self, req_id: str) -> list[str]:
        """Hazards a requirement is a risk control for."""
        return [h.id for h in self.hazards if req_id in h.controls]


@cache
def _definitions(path: Path, mtime: float) -> frozenset[str]:
    """Top-level names and ``Class.method`` names defined in a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
            if isinstance(node, ast.ClassDef):
                names |= {
                    f"{node.name}.{n.name}"
                    for n in node.body
                    if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                }
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return frozenset(names)


def _reference_problem(root: Path, ref: str, what: str) -> str | None:
    """Why ``ref`` (``path`` or ``path::name``) does not resolve, or ``None``."""
    path, _, name = ref.partition("::")
    file = root / path
    if not file.exists():
        return f"{what} {ref}: {path} does not exist"
    if name:
        if file.suffix != ".py" or not file.is_file():
            return f"{what} {ref}: a name can only be given in a Python file"
        if name not in _definitions(file, file.stat().st_mtime):
            return f"{what} {ref}: {name} is not defined in {path}"
    return None


def load_trace(root: Path) -> Trace:
    """Read and validate requirements and hazards under ``root``; raise ``ValueError`` listing
    every broken reference."""
    reqs = RequirementFile.model_validate(load_yaml(root / REQUIREMENTS_FILE)).requirements
    hz = HazardFile.model_validate(load_yaml(root / HAZARDS_FILE))
    trace = Trace(reqs, hz.hazards, hz.policy)
    problems = check_trace(trace, root)
    if problems:
        raise ValueError("traceability check failed:\n- " + "\n- ".join(problems))
    return trace


def check_trace(trace: Trace, root: Path) -> list[str]:
    """Every broken reference or missing link (empty when consistent)."""
    problems = []
    for kind, ids in (("requirement", [r.id for r in trace.requirements]),
                      ("hazard", [h.id for h in trace.hazards])):  # fmt: skip
        problems += [f"{kind} {i} is defined twice" for i, n in Counter(ids).items() if n > 1]
    req_ids = {r.id for r in trace.requirements}
    for r in trace.requirements:
        for ref in r.code:
            problems.append(_reference_problem(root, ref, f"{r.id} code"))
        for ref in r.tests:
            if not ref.startswith("tests/") or "::" not in ref:
                problems.append(f"{r.id} test {ref}: expected tests/<file>.py::<test>")
            else:
                problems.append(_reference_problem(root, ref, f"{r.id} test"))
        for ref in r.reports:
            problems.append(_reference_problem(root, ref, f"{r.id} report"))
    for h in trace.hazards:
        for c in h.controls:
            if c not in req_ids:
                problems.append(f"hazard {h.id}: control {c} is not a requirement")
        for ref in h.evidence:
            problems.append(_reference_problem(root, ref, f"hazard {h.id} evidence"))
    return [p for p in problems if p]
