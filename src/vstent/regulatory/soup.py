"""SOUP list and software bill of materials from ``uv.lock`` (IEC 62304 §5.3.3, §8.1.2).

SOUP (software of unknown provenance) is every third-party component the product runs. The
locked dependency graph is walked from the project's own entry for each target platform, so
the list is exactly what ``uv sync`` installs there. Curated facts per direct dependency
(purpose, what the product relies on, the hazards it can contribute to, the review of its
published anomalies) live in ``docs/regulatory/soup.yaml``.

The check: every direct dependency has a SOUP entry, every entry is a direct dependency, and
every third-party module imported anywhere in ``src/vstent`` belongs to a SOUP entry.
"""

import ast
import hashlib
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from vstent.config import StrictModel, load_yaml

LOCK_FILE = "uv.lock"
PYTHON_VERSION_FILE = ".python-version"
SOURCE_DIR = Path("src/vstent")
SOUP_FILE = Path("docs/regulatory/soup.yaml")

# Target platforms of the release (CPython from .python-version on 64-bit Windows and Linux):
# PEP 508 environment markers for each.
PLATFORMS = {
    "win32": {"sys_platform": "win32", "platform_system": "Windows", "os_name": "nt",
              "platform_machine": "AMD64"},
    "linux": {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix",
              "platform_machine": "x86_64"},
}  # fmt: skip
SHIPPED_EXTRAS = ("imaging", "app")  # installed with the product
TOOL_EXTRAS = ("dev",)  # development and documentation tools, not shipped


class SoupItem(StrictModel):
    """Curated facts about one direct dependency (one entry of ``soup.yaml``)."""

    name: str
    imports: list[str] = Field(default_factory=list)  # module names; [] if never imported
    purpose: str
    relied_on: str  # functional / performance requirements on the SOUP (IEC 62304 §5.3.3)
    hazards: list[str] = Field(default_factory=list)
    anomalies: str  # status of the published-anomaly review (IEC 62304 §7.1.2–7.1.3)


class PlatformItem(StrictModel):
    """SOUP outside the lock: interpreter, operating system, browser."""

    name: str
    version: str
    purpose: str
    notes: str = ""


class SoupFile(StrictModel):
    soup: list[SoupItem]
    platform: list[PlatformItem] = Field(default_factory=list)


@dataclass(frozen=True)
class Locked:
    """One locked package in the release closure."""

    name: str
    version: str
    source: str
    platforms: tuple[str, ...]
    direct: bool
    scope: str  # "runtime" (shipped) or "development"


def normalise(name: str) -> str:
    """PEP 503 normalised distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def python_version(root: Path) -> str:
    """The release interpreter version (``.python-version``), as major.minor."""
    return (root / PYTHON_VERSION_FILE).read_text(encoding="utf-8").strip()


def environment(platform: str, python: str) -> dict[str, str]:
    """PEP 508 marker environment of a target platform."""
    full = python if python.count(".") >= 2 else f"{python}.0"
    return {
        **PLATFORMS[platform],
        "python_version": ".".join(full.split(".")[:2]),
        "python_full_version": full,
        "implementation_name": "cpython",
        "implementation_version": full,
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_version": "",
        "extra": "",
    }


def _marker_ok(marker: str | None, env: dict[str, str]) -> bool:
    from packaging.markers import Marker  # a dev tool: only `vstent regulatory` needs it

    return marker is None or Marker(marker).evaluate(env)


def _source(entry: dict[str, Any]) -> str:
    src = entry.get("source", {})
    for key in ("registry", "url", "git", "path", "editable", "virtual"):
        if key in src:
            return str(src[key])
    return ""


class Lock:
    """A parsed ``uv.lock`` (lock format version 1)."""

    def __init__(self, text: str):
        data = tomllib.loads(text)
        if data.get("version") != 1:
            raise ValueError(f"unsupported uv.lock version {data.get('version')}")
        self.sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.packages = data["package"]
        self.by_name: dict[str, list[dict[str, Any]]] = {}
        for p in self.packages:
            self.by_name.setdefault(normalise(p["name"]), []).append(p)
        roots = [p for p in self.packages if p.get("source", {}).get("editable") == "."]
        if len(roots) != 1:
            raise ValueError("uv.lock: expected exactly one editable project entry")
        self.root = roots[0]

    @classmethod
    def read(cls, root: Path) -> "Lock":
        return cls((root / LOCK_FILE).read_text(encoding="utf-8"))

    def _resolve(self, dep: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
        cands = self.by_name.get(normalise(dep["name"]), [])
        if "version" in dep:
            cands = [c for c in cands if c["version"] == dep["version"]]
        if "source" in dep:
            cands = [c for c in cands if c.get("source") == dep["source"]]
        if len(cands) > 1:
            cands = [
                c for c in cands
                if any(_marker_ok(m, env) for m in c.get("resolution-markers", [None]))
            ]  # fmt: skip
        if len(cands) != 1:
            raise ValueError(f"uv.lock: cannot resolve {dep['name']} ({len(cands)} candidates)")
        return cands[0]

    def direct(self, extras: tuple[str, ...]) -> set[str]:
        """Normalised names the project itself declares (main dependencies + ``extras``)."""
        deps = list(self.root.get("dependencies", []))
        for e in extras:
            deps += self.root.get("optional-dependencies", {}).get(e, [])
        return {normalise(d["name"]) for d in deps}

    def closure(self, env: dict[str, str], extras: tuple[str, ...]) -> dict[tuple, dict]:
        """Every package installed for the project with ``extras`` in ``env``, keyed by
        (name, version, source)."""
        seen: dict[tuple, set[str]] = {}
        out: dict[tuple, dict] = {}
        stack = [(self.root, set(extras))]
        while stack:
            entry, want = stack.pop()
            key = (normalise(entry["name"]), entry["version"], _source(entry))
            done = seen.get(key)
            new = want - done if done is not None else want
            if done is not None and not new:
                continue
            deps = [] if done is not None else list(entry.get("dependencies", []))
            for e in sorted(new):
                deps += entry.get("optional-dependencies", {}).get(e, [])
            seen[key] = (done or set()) | want
            if entry is not self.root:
                out[key] = entry
            for d in deps:
                if _marker_ok(d.get("marker"), env):
                    stack.append((self._resolve(d, env), set(d.get("extra", []))))
        return out


def release_packages(lock: Lock, python: str) -> list[Locked]:
    """The shipped closure and the development-only packages over all target platforms."""
    direct_rt, direct_dev = lock.direct(()) | lock.direct(SHIPPED_EXTRAS), lock.direct(TOOL_EXTRAS)
    rows: dict[tuple, dict[str, Any]] = {}
    for plat in PLATFORMS:
        env = environment(plat, python)
        runtime = lock.closure(env, SHIPPED_EXTRAS)
        everything = lock.closure(env, SHIPPED_EXTRAS + TOOL_EXTRAS)
        for key in everything:
            row = rows.setdefault(key, {"platforms": set(), "scope": "development"})
            row["platforms"].add(plat)
            if key in runtime:
                row["scope"] = "runtime"
    out = []
    for (name, version, source), row in rows.items():
        direct = name in (direct_rt if row["scope"] == "runtime" else direct_rt | direct_dev)
        out.append(
            Locked(name, version, source, tuple(sorted(row["platforms"])), direct, row["scope"])
        )
    return sorted(out, key=lambda p: (p.scope != "runtime", p.name, p.version))


def third_party_imports(root: Path) -> dict[str, set[str]]:
    """Top-level third-party modules imported anywhere under ``src/vstent`` → the importing
    files (relative paths)."""
    skip = set(sys.stdlib_module_names) | {"vstent", "__future__"}
    found: dict[str, set[str]] = {}
    for path in sorted((root / SOURCE_DIR).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for n in names:
                top = n.split(".")[0]
                if top not in skip:
                    found.setdefault(top, set()).add(path.relative_to(root).as_posix())
    return found


def load_soup(root: Path) -> SoupFile:
    return SoupFile.model_validate(load_yaml(root / SOUP_FILE))


def check_soup(soup: SoupFile, lock: Lock, imports: dict[str, set[str]]) -> list[str]:
    """Problems with the SOUP list (empty when consistent)."""
    problems = []
    declared = lock.direct(()) | lock.direct(SHIPPED_EXTRAS) | lock.direct(TOOL_EXTRAS)
    names = [normalise(s.name) for s in soup.soup]
    for n in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"{SOUP_FILE}: {n} is listed twice")
    for n in sorted(declared - set(names)):
        problems.append(f"{SOUP_FILE}: direct dependency {n} has no entry")
    for n in sorted(set(names) - declared):
        problems.append(f"{SOUP_FILE}: {n} is not a direct dependency in uv.lock")
    owner = {m: s.name for s in soup.soup for m in s.imports}
    for module, files in sorted(imports.items()):
        if module not in owner:
            problems.append(
                f"{module} is imported ({', '.join(sorted(files))}) but belongs to no SOUP "
                "entry: declare it in pyproject.toml and soup.yaml"
            )
    return problems


def software_items(files: set[str]) -> list[str]:
    """Software items (sub-packages of ``vstent``; top-level modules by name) of files."""
    out = set()
    for f in files:
        parts = Path(f).relative_to(SOURCE_DIR).parts
        out.add(parts[0] if len(parts) > 1 else Path(parts[0]).stem)
    return sorted(out)


def _purl(p: Locked) -> str:
    purl = f"pkg:pypi/{p.name}@{p.version.replace('+', '%2B')}"
    if p.source and "pypi.org" not in p.source:
        purl += f"?repository_url={p.source}"
    return purl


def sbom(packages: list[Locked], lock: Lock, python: str) -> dict[str, Any]:
    """CycloneDX 1.5 SBOM of the locked closure (deterministic: no timestamp, no serial)."""
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": f"vstent@{lock.root['version']}",
                "name": "vstent",
                "version": lock.root["version"],
                "description": "Virtual stenting planner: research prototype, not a medical "
                "device, not for clinical use.",
            },
            "properties": [
                {"name": "vstent:uv.lock:sha256", "value": lock.sha256},
                {"name": "vstent:python", "value": python},
                {"name": "vstent:platforms", "value": ", ".join(PLATFORMS)},
            ],
        },
        "components": [
            {
                "type": "library",
                "bom-ref": _purl(p),
                "name": p.name,
                "version": p.version,
                "purl": _purl(p),
                "scope": "required" if p.scope == "runtime" else "excluded",
                "properties": [
                    {"name": "vstent:platforms", "value": ", ".join(p.platforms)},
                    {"name": "vstent:direct", "value": str(p.direct).lower()},
                ],
            }
            for p in packages
        ],
    }


def sbom_json(packages: list[Locked], lock: Lock, python: str) -> str:
    return json.dumps(sbom(packages, lock, python), indent=2, ensure_ascii=False) + "\n"
