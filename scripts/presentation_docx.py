"""The presentation brief as a Word file with its figures embedded (one file to upload).

Converts docs/presentation/vstent_presentation_brief.md (headings, paragraphs, quotes, bullet
and numbered lists, tables, **bold**, *italic*, `code`) and places each figure named in a
"Figure(s):" line right after that line.

    uv run --with python-docx python scripts/presentation_docx.py
"""

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

SRC = Path("docs/presentation/vstent_presentation_brief.md")
OUT = SRC.with_suffix(".docx")
FIG_DIR = SRC.parent
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
FIGURE = re.compile(r"figures/[\w\-]+\.png")


def runs(par, text: str) -> None:
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        elif part.startswith("*") and len(part) > 2:
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def shade(par, fill: str) -> None:
    ppr = par._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    ppr.append(shd)


def rule(doc) -> None:
    par = doc.add_paragraph()
    ppr = par._p.get_or_add_pPr()
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"), ("w:color", "C3C2B7")):
        bottom.set(qn(k), v)
    border.append(bottom)
    ppr.append(border)


def table(doc, rows: list[str]) -> None:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(re.fullmatch(r":?-{3,}:?", x) for x in c)]
    t = doc.add_table(rows=len(cells), cols=len(cells[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(cells):
        for j, text in enumerate(row[: len(cells[0])]):
            cell = t.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            runs(par, text)
            for r in par.runs:
                r.font.size = Pt(9)
                if i == 0:
                    r.bold = True
            if i == 0:
                shade(par, "E8EEF8")
    doc.add_paragraph()


def figures(doc, line: str, placed: set) -> None:
    for name in FIGURE.findall(line):
        path = FIG_DIR / name
        if name in placed or not path.exists():
            continue
        placed.add(name)
        doc.add_picture(str(path), width=Inches(6.3))
        cap = doc.add_paragraph()
        r = cap.add_run(name.split("/")[-1])
        r.italic = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)


def main() -> None:
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.8)
        s.top_margin = s.bottom_margin = Inches(0.8)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    lines = SRC.read_text(encoding="utf-8").splitlines()
    placed: set[str] = set()
    para: list[str] = []
    in_appendix_b = False

    def flush():
        if para:
            runs(doc.add_paragraph(), " ".join(x.strip() for x in para))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            flush()
            i += 1
            continue
        if s.startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            table(doc, block)
            continue
        if s == "---":
            flush()
            rule(doc)
        elif s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            text = s[level:].strip().replace("`", "")
            in_appendix_b = text.startswith("Appendix B")
            doc.add_heading(text, level=0 if level == 1 else min(level - 1, 3))
        elif s.startswith(">"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            par = doc.add_paragraph()
            runs(par, " ".join(block))
            shade(par, "FFF4E5")
            continue
        elif re.match(r"^\s*(- |\d+\. )", line):
            flush()
            indent = len(line) - len(line.lstrip())
            numbered = bool(re.match(r"^\s*\d+\. ", line))
            text = re.sub(r"^\s*(- |\d+\. )", "", line)
            i += 1
            while (i < len(lines) and lines[i].strip() and not re.match(r"^\s*(- |\d+\. |\|)",
                                                                        lines[i])
                   and len(lines[i]) - len(lines[i].lstrip()) > indent):  # fmt: skip
                text += " " + lines[i].strip()
                i += 1
            style = "List Number" if numbered else "List Bullet"
            if indent >= 2:
                style += " 2"
            runs(doc.add_paragraph(style=style), text)
            if not in_appendix_b and ("Figure:" in text or "Figures:" in text):
                figures(doc, text, placed)
            continue
        else:
            para.append(s)
        i += 1
    flush()
    doc.save(OUT)
    print(f"wrote {OUT} with {len(placed)} figures")


if __name__ == "__main__":
    main()
