"""Convert ../README.md into build/readme.tex for the ICML-style Appendix (full text/tables/code, publication typography).

- Extracts two distinct frames from docs/images/browser-demo.gif for Figure 2 in Section 2:
    * build/browser-demo-cards.png   (frame 63: 3-item product cards & prices)
    * build/browser-demo-receipt.png (frame 67: verified budget receipt & sources)
- Keeps 100% of README.md's prose, headings, tables, links, and code blocks in Appendix A, while replacing
  the two inline images (already shown as Figures 1 and 2 in Section 2) with crisp cross-references so no
  image is duplicated in the PDF.
- Formats tables with bold tinted headers and exact full-width booktabs columns, wraps code blocks in clean
  breakable tcolorbox frames (no broken borders across page breaks), and strips raw Markdown horizontal rules.

    python3 readme_to_tex.py            # needs pandoc and Pillow
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "build"
BLOB = "https://github.com/atlasmarkz/tacc-agentforge/blob/main/"


def gif_frames():
    from PIL import Image, ImageSequence
    frames = [f.copy().convert("RGB") for f in ImageSequence.Iterator(Image.open(ROOT / "docs/images/browser-demo.gif"))]
    cards_idx = 63 if len(frames) > 63 else 0
    receipt_idx = 67 if len(frames) > 67 else -1
    frames[cards_idx].save(BUILD / "browser-demo-cards.png")
    frames[receipt_idx].save(BUILD / "browser-demo-receipt.png")


def markdown():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    head, sep, rest = text.partition("\nTACC AgentForge is the course project")
    if not sep:
        raise SystemExit("README header changed; update readme_to_tex.py")
    tagline = re.search(r"<strong>(.*?)</strong>", head).group(1)
    rest = sep.lstrip("\n") + rest

    # Replace the two inline README images with cross-references to Figures 1 and 2 in Section 2 (avoids duplicate images).
    rest = re.sub(
        r'<p align="center">\s*<img src="docs/images/browser-demo\.gif"[^>]*>\s*</p>',
        r'> **[README Figure — Live TACC Browser Demo]** *Shown in Figure 2 (Section 2) to avoid duplicating figures in this PDF.*',
        rest,
    )
    rest = re.sub(
        r'<p align="center">\s*<img src="docs/images/architecture\.png"[^>]*>\s*</p>',
        r'> **[README Figure — Dual-Branch System Architecture]** *Shown in Figure 1 (Section 2) to avoid duplicating figures in this PDF.*',
        rest,
    )
    # Remove raw horizontal rules ("---") so clean LaTeX section hierarchy provides the visual breaks.
    rest = re.sub(r"(?m)^\s*---\s*$", "", rest)

    md = f"# Overview: TACC AgentForge\n\n*{tagline}*\n\n" + rest

    def link(m):
        label, target = m.group(1), m.group(2)
        if target.startswith(("http://", "https://", "#", "build/", "mailto:")) or m.group(0).startswith("!"):
            return m.group(0)
        return f"[{label}]({BLOB}{target})"
    return re.sub(r"!?\[([^\]]*)\]\(([^)\s]+)\)", link, md)


def _bold_header_row(row_text: str) -> str:
    """Wrap each cell in a booktabs header row with \\textbf{...} for ICML-style tables."""
    # Remove trailing \\ if present, bold each &-separated cell, re-append \\
    m = re.match(r"^(.*?)(\\\\(?:\[[^\]]*\])?\s*)$", row_text, flags=re.S)
    if not m:
        return row_text
    cells = [c.strip() for c in m.group(1).split("&")]
    bolded = [f"\\textbf{{{c}}}" if c and not c.startswith("\\textbf") else c for c in cells]
    return " & ".join(bolded) + " " + m.group(2)


def _format_table(m: re.Match) -> str:
    spec, body = m.group(1), m.group(2)
    # Strip pandoc longtable headers/footers while keeping header + data rows.
    lines = [
        l for l in body.split("\n")
        if l.strip() not in ("\\endhead", "\\endfirsthead", "\\endlastfoot", "\\endfoot", "\\bottomrule\\noalign{}")
    ]
    # Count columns and compute proportional widths that fill 100% of \linewidth cleanly.
    if "p{" not in spec:
        cols = len(re.findall(r"[lrc]", spec.replace("@{}", "")))
        rows = [r for r in re.split(r"\\\\", "\n".join(lines)) if "&" in r]
        longest, word = [1] * cols, [1] * cols
        for row in rows:
            for i, cell in enumerate(row.split("&")[:cols]):
                plain = re.sub(r"\\[a-zA-Z]+|[{}]", "", cell).strip()
                longest[i] = max(longest[i], len(plain))
                word[i] = max([word[i]] + [len(w) for w in re.split(r"[\s/]+", plain)])
        floor = [max(0.07, (w + 2) / 108) for w in word]
        share = [max(f, n / sum(longest)) for f, n in zip(floor, longest)]
        share = [s / sum(share) for s in share]
        # Exact width accounting for inter-column tabcolsep only (outer @{} removes outer padding).
        tabsep_count = max(0, 2 * (cols - 1))
        spec = "@{}" + "".join(
            f">{{\\raggedright\\arraybackslash}}p{{(\\linewidth - {tabsep_count}\\tabcolsep) * \\real{{{s:.4f}}}}}"
            for s in share
        ) + "@{}"
    else:
        # Pandoc emitted \real{...} column widths; normalize outer padding cleanly.
        spec = spec.strip()

    # Style the header block between \toprule and \midrule with subtle RowTint + bold column names.
    joined = "\n".join(lines)
    hm = re.search(r"(\\toprule(?:\\noalign\{\})?\s*\n)(.*?)(\n\s*\\midrule)", joined, flags=re.S)
    if hm:
        header_block = hm.group(2).strip()
        # Handle multi-line pandoc headers (often wrapped across 2 lines with \begin{minipage} or plain text).
        if "\\begin{minipage}" in header_block:
            styled_header = re.sub(
                r"(\\begin\{minipage\}\[b\]\{\\linewidth\}\\raggedright\s*)(.*?)(\s*\\end\{minipage\})",
                lambda cm: cm.group(1) + f"\\textbf{{{cm.group(2).strip()}}}" + cm.group(3),
                header_block,
                flags=re.S,
            )
            styled_header = "\\rowcolor{RowTint}" + styled_header
        else:
            styled_header = "\\rowcolor{RowTint}" + _bold_header_row(header_block)
        joined = joined[:hm.start(2)] + styled_header + joined[hm.end(2):]

    # Add gentle vertical breathing room between multi-line body rows.
    joined = re.sub(r"\\addlinespace\b", r"\\addlinespace[2pt]", joined)
    return (
        "\\par\\medskip\\noindent{\\footnotesize\\renewcommand{\\arraystretch}{1.18}\\setlength{\\tabcolsep}{6pt}"
        f"\\begin{{tabular}}{{{spec}}}\n"
        + joined.rstrip()
        + "\n\\bottomrule\n\\end{tabular}}\\par\\medskip\n"
    )


def to_latex(md):
    done = subprocess.run(
        ["pandoc", "-f", "gfm", "-t", "latex", "--syntax-highlighting=none", "--wrap=preserve", "--shift-heading-level-by=1"],
        input=md, capture_output=True, text=True, check=True,
    )
    tex = done.stdout

    # Remove any stray horizontal rule blocks if present.
    tex = re.sub(r"\\begin\{center\}\\rule\{[^}]*\}\{[^}]*\}\\end\{center\}\s*", "", tex)

    # Code blocks: sleek breakable tcolorbox with soft paper tint and crisp 7.8pt monospace (fits 96+ cols cleanly).
    tex = tex.replace(
        "\\begin{verbatim}",
        "\\begin{codebox}\n\\begin{Verbatim}[breaklines,breakanywhere,breaksymbolleft={},fontsize=\\fontsize{7.7}{9.4}\\selectfont]",
    )
    tex = tex.replace(
        "\\end{verbatim}",
        "\\end{Verbatim}\n\\end{codebox}",
    )

    # Cross-reference callouts from markdown blockquotes.
    tex = tex.replace("\\begin{quote}", "\\begin{readmeNote}")
    tex = tex.replace("\\end{quote}", "\\end{readmeNote}")

    # Allow clean line breaks after "/" and "_" in inline monospace paths and identifiers.
    tex = re.sub(
        r"\\texttt\{((?:\\[{}]|[^{}])*)\}",
        lambda m: "\\texttt{" + re.sub(r"(?<!\\)/", "/\\\\allowbreak{}", m.group(1)).replace("\\_", "\\_\\allowbreak{}") + "}",
        tex,
    )

    # Tables: full-width booktabs with bold tinted headers.
    tex = re.sub(r"\\begin\{longtable\}\[\]\{(.*?)\}\n(.*?)\\end\{longtable\}", _format_table, tex, flags=re.S)
    return tex


def main():
    BUILD.mkdir(exist_ok=True)
    gif_frames()
    (BUILD / "readme.tex").write_text(to_latex(markdown()), encoding="utf-8")
    print("wrote", BUILD / "readme.tex")


if __name__ == "__main__":
    main()
