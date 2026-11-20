"""Convert ../README.md into build/readme.tex for the report's appendix (full text/tables/code, zero duplicate images).

- Extracts two distinct frames from docs/images/browser-demo.gif for Figure 2 in Section 2:
    * build/browser-demo-cards.png   (frame 63: 3-item product cards & prices)
    * build/browser-demo-receipt.png (frame 67: verified budget receipt & sources)
- Keeps 100% of README.md's prose, headings, tables, links, and code blocks in Appendix A, while replacing
  the two inline images (already shown as Figures 1 and 2 in Section 2) with crisp cross-references so no
  image is duplicated in the PDF.
- Relative links point to files on GitHub so every link stays clickable in the PDF.
- Uses pandoc -> LaTeX with wrapped code blocks (fvextra) and clean proportional booktabs tables.

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
        r'> **[README Figure — Live TACC Browser Demo]** *Shown in Figure 2 (Section 2) to avoid duplicating images in this PDF.*',
        rest,
    )
    rest = re.sub(
        r'<p align="center">\s*<img src="docs/images/architecture\.png"[^>]*>\s*</p>',
        r'> **[README Figure — Evaluated Deployment Architecture]** *Shown in Figure 1 (Section 2) to avoid duplicating images in this PDF.*',
        rest,
    )

    md = f"# TACC AgentForge\n\n**{tagline}**\n\n" + rest

    def link(m):
        label, target = m.group(1), m.group(2)
        if target.startswith(("http://", "https://", "#", "build/", "mailto:")) or m.group(0).startswith("!"):
            return m.group(0)
        return f"[{label}]({BLOB}{target})"
    return re.sub(r"!?\[([^\]]*)\]\(([^)\s]+)\)", link, md)


def to_latex(md):
    done = subprocess.run(["pandoc", "-f", "gfm", "-t", "latex", "--syntax-highlighting=none", "--wrap=preserve",
                           "--shift-heading-level-by=1"], input=md, capture_output=True, text=True, check=True)
    tex = done.stdout
    # Code blocks: clean framed monospace with subtle tint and line wrapping.
    tex = tex.replace(
        "\\begin{verbatim}",
        "\\begin{Verbatim}[breaklines,breakanywhere,fontsize=\\footnotesize,frame=single,framerule=0.4pt,rulecolor=\\color{BorderGray},framesep=4pt]",
    )
    tex = tex.replace("\\end{verbatim}", "\\end{Verbatim}")
    # Cross-reference callouts from markdown blockquotes.
    tex = tex.replace("\\begin{quote}", "\\begin{readmeNote}")
    tex = tex.replace("\\end{quote}", "\\end{readmeNote}")
    # Code font: allow line breaks after "/" and "_" in paths and file names.
    tex = re.sub(r"\\texttt\{((?:\\[{}]|[^{}])*)\}",
                 lambda m: "\\texttt{" + re.sub(r"(?<!\\)/", "/\\\\allowbreak{}", m.group(1)).replace("\\_", "\\_\\allowbreak{}") + "}", tex)
    # Tables: longtable -> proportional tabular with booktabs rules.
    def table(m):
        spec, body = m.group(1), m.group(2)
        if "p{" not in spec:
            cols = len(re.findall(r"[lrc]", spec.replace("@{}", "")))
            rows = [r for r in re.split(r"\\\\", body) if "&" in r]
            longest, word = [1] * cols, [1] * cols
            for row in rows:
                for i, cell in enumerate(row.split("&")[:cols]):
                    plain = re.sub(r"\\[a-zA-Z]+|[{}]", "", cell).strip()
                    longest[i] = max(longest[i], len(plain))
                    word[i] = max([word[i]] + [len(w) for w in re.split(r"[\s/]+", plain)])
            floor = [max(0.08, (w + 2) / 100) for w in word]
            share = [max(f, n / sum(longest)) for f, n in zip(floor, longest)]
            share = [s / sum(share) for s in share]
            spec = "@{}" + "".join(
                f">{{\\raggedright\\arraybackslash}}p{{(\\linewidth - {2 * cols}\\tabcolsep) * \\real{{{s:.4f}}}}}"
                for s in share) + "@{}"
        lines = [l for l in body.split("\n") if l.strip() not in ("\\endhead", "\\endfirsthead", "\\endlastfoot",
                                                                     "\\endfoot", "\\bottomrule\\noalign{}")]
        return ("\\par\\smallskip\\noindent{\\footnotesize\\renewcommand{\\arraystretch}{1.15}\\begin{tabular}{" + spec + "}\n" + "\n".join(lines).rstrip()
                + "\n\\bottomrule\n\\end{tabular}}\\par\\smallskip\n")
    tex = re.sub(r"\\begin\{longtable\}\[\]\{(.*?)\}\n(.*?)\\end\{longtable\}", table, tex, flags=re.S)
    return tex


def main():
    BUILD.mkdir(exist_ok=True)
    gif_frames()
    (BUILD / "readme.tex").write_text(to_latex(markdown()), encoding="utf-8")
    print("wrote", BUILD / "readme.tex")


if __name__ == "__main__":
    main()
