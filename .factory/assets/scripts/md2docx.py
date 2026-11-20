#!/usr/bin/env python3
"""Renderiza markdown (subconjunto controlado) a un .docx con estilo de guía.
Uso: md2docx.py <input.md> <output.docx> "<Subtitulo / audiencia>" [appendix.md "<Titulo anexo>"]
"""
import os, sys, re, datetime
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

CALLOUT_COLORS = {
    "Nota": "DDEBF7", "⚠ Cuidado": "FCE4D6", "Cuidado": "FCE4D6",
    "✅ Hecho": "E2EFDA", "Hecho": "E2EFDA",
    "👉 Para tu técnico": "FFF2CC", "Para tu técnico": "FFF2CC",
}
DEFAULT_CALLOUT = "F2F2F2"

INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+?`)")

def shade(el, hexcolor):
    sh = OxmlElement('w:shd'); sh.set(qn('w:val'), 'clear'); sh.set(qn('w:fill'), hexcolor)
    el.append(sh)

def set_cell_bg(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); shade(tcPr, hexcolor)

def add_runs(p, text):
    """Renderiza **negrita** y `código` dentro de un párrafo."""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = p.add_run(part[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0xB0, 0x30, 0x60)
        else:
            p.add_run(part)

def hr(doc):
    p = doc.add_paragraph(); pPr = p._p.get_or_add_pPr()
    pb = OxmlElement('w:pBdr'); bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single'); bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1'); bottom.set(qn('w:color'), 'BBBBBB')
    pb.append(bottom); pPr.append(pb)

def add_table(doc, rows):
    header, body = rows[0], rows[1:]
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.paragraphs[0].text = ""
        add_runs(c.paragraphs[0], h.strip())
        for run in c.paragraphs[0].runs: run.bold = True
        set_cell_bg(c, "4472C4")
        for run in c.paragraphs[0].runs: run.font.color.rgb = RGBColor(0xFF,0xFF,0xFF)
    for r in body:
        cells = t.add_row().cells
        for i in range(len(header)):
            cells[i].paragraphs[0].text = ""
            add_runs(cells[i].paragraphs[0], (r[i].strip() if i < len(r) else ""))

def flush_callout(doc, lines):
    if not lines: return
    # detect label on first line
    fill = DEFAULT_CALLOUT; first = lines[0]
    m = re.match(r"\*\*(.+?):\*\*\s*(.*)", first)
    label = None
    if m:
        label = m.group(1).strip()
        fill = CALLOUT_COLORS.get(label, DEFAULT_CALLOUT)
    t = doc.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = t.rows[0].cells[0]; set_cell_bg(cell, fill)
    cell.paragraphs[0].text = ""
    if m:
        p = cell.paragraphs[0]; r = p.add_run(m.group(1) + ": "); r.bold = True
        add_runs(p, m.group(2))
        rest = lines[1:]
    else:
        add_runs(cell.paragraphs[0], first); rest = lines[1:]
    for ln in rest:
        p = cell.add_paragraph(); add_runs(p, ln)
    doc.add_paragraph()

def add_image(doc, alt, src, base_dir):
    path = src if os.path.isabs(src) else os.path.normpath(os.path.join(base_dir, src))
    try:
        doc.add_picture(path, width=Inches(6.3))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    except Exception:
        p = doc.add_paragraph(); r = p.add_run("[imagen no encontrada: %s]" % src)
        r.italic = True; r.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
    if alt:
        cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rc = cap.add_run(alt); rc.italic = True; rc.font.size = Pt(9)
        rc.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

def render_md(doc, text, base_heading=0, base_dir="."):
    lines = text.split("\n")
    i = 0; callout = []; table_block = []; code_block = []; in_code = False
    def flush_table():
        nonlocal table_block
        if table_block:
            rows = []
            for rl in table_block:
                if re.match(r"^\s*\|?\s*[-:]+\s*(\|\s*[-:]+\s*)+\|?\s*$", rl):  # separator
                    continue
                inner = rl.strip()
                if inner.startswith("|"): inner = inner[1:]
                if inner.endswith("|"):   inner = inner[:-1]
                cells = [c.replace("\\|", "|") for c in re.split(r"(?<!\\)\|", inner)]
                rows.append(cells)
            if rows: add_table(doc, rows)
            table_block = []
    def flush_code():
        nonlocal code_block
        if code_block:
            for cl in code_block:
                p = doc.add_paragraph(); pPr = p._p.get_or_add_pPr(); shade(pPr, "F4F4F4")
                r = p.add_run(cl or " "); r.font.name = "Consolas"; r.font.size = Pt(9)
            doc.add_paragraph(); code_block = []
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            if in_code: flush_code(); in_code = False
            else: in_code = True
            i += 1; continue
        if in_code:
            code_block.append(ln); i += 1; continue
        if ln.startswith("> "):
            flush_table(); callout.append(ln[2:]); i += 1; continue
        else:
            if callout: flush_callout(doc, callout); callout = []
        if "|" in ln and ln.strip().startswith("|"):
            table_block.append(ln); i += 1; continue
        else:
            flush_table()
        s = ln.rstrip()
        if not s.strip():
            i += 1; continue
        if s.strip() == "---":
            hr(doc); i += 1; continue
        m = re.match(r"^(#{1,6})\s+(.*)", s)
        if m:
            lvl = min(max(len(m.group(1)) + base_heading, 1), 9)
            h = doc.add_heading(level=lvl)
            add_runs(h, m.group(2))
            i += 1; continue
        m = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", s)
        if m:
            add_image(doc, m.group(1).strip(), m.group(2).strip(), base_dir); i += 1; continue
        m = re.match(r"^\[Captura:\s*(.*)\]\s*$", s)
        if m:
            p = doc.add_paragraph(); r = p.add_run("🖼  [Captura: " + m.group(1) + "]")
            r.italic = True; r.font.color.rgb = RGBColor(0x88,0x88,0x88); i += 1; continue
        m = re.match(r"^(\s*)-\s+(.*)", s)
        if m:
            indent = len(m.group(1)); style = "List Bullet 2" if indent >= 2 else "List Bullet"
            try: p = doc.add_paragraph(style=style)
            except KeyError: p = doc.add_paragraph(style="List Bullet")
            add_runs(p, m.group(2)); i += 1; continue
        m = re.match(r"^\s*\d+\.\s+(.*)", s)
        if m:
            p = doc.add_paragraph(style="List Number"); add_runs(p, m.group(1)); i += 1; continue
        p = doc.add_paragraph(); add_runs(p, s); i += 1
    if callout: flush_callout(doc, callout)
    flush_table();
    if in_code: flush_code()

def strip_frontmatter(text):
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            nl = text.find("\n", end + 1)
            return text[nl + 1:] if nl != -1 else ""
    return text

def build(inp, outp, subtitle, appendix=None, appendix_title=None):
    base_dir = os.path.dirname(os.path.abspath(inp))
    raw = strip_frontmatter(open(inp, encoding="utf-8").read())
    lines = raw.split("\n")
    title = "Guía"
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip(); raw = "\n".join(lines[1:])
    doc = Document()
    style = doc.styles["Normal"]; style.font.name = "Calibri"; style.font.size = Pt(11)
    # cover
    doc.add_paragraph()
    t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(title); r.bold = True; r.font.size = Pt(26); r.font.color.rgb = RGBColor(0x2F,0x54,0x96)
    s = doc.add_paragraph(); s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs = s.add_run(subtitle); rs.font.size = Pt(14); rs.font.color.rgb = RGBColor(0x60,0x60,0x60)
    d = doc.add_paragraph(); d.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rd = d.add_run("Software Factory · " + datetime.date.today().strftime("%B %Y")); rd.italic = True
    doc.add_page_break()
    render_md(doc, raw, base_heading=-1, base_dir=base_dir)
    if appendix:
        doc.add_page_break()
        h = doc.add_heading(level=1); add_runs(h, appendix_title or "Anexo")
        render_md(doc, strip_frontmatter(open(appendix, encoding="utf-8").read()),
                  base_heading=0, base_dir=os.path.dirname(os.path.abspath(appendix)))
    doc.save(outp)
    print("OK ->", outp)

if __name__ == "__main__":
    a = sys.argv
    build(a[1], a[2], a[3],
          appendix=(a[4] if len(a) > 4 else None),
          appendix_title=(a[5] if len(a) > 5 else None))
