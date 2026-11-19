"""Convert ../README.md into build/readme.tex for the report's appendix (full contents, black-and-white layout).

- The HTML header (logo, tagline, links, animated GIF) becomes Markdown; the GIF is shown as its last frame.
- Relative links point to the files on GitHub, so they stay clickable in the PDF.
- pandoc converts GitHub Markdown to LaTeX without syntax colors; code blocks wrap long lines (fvextra), and
  tables become non-breaking booktabs tables.

    python3 readme_to_tex.py            # needs pandoc and Pillow
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "build"
BLOB = "https://github.com/atlasmarkz/tacc-agentforge/blob/main/"


def gif_frame():
    from PIL import Image, ImageSequence
    frames = [f.copy().convert("RGB") for f in ImageSequence.Iterator(Image.open(ROOT / "docs/images/browser-demo.gif"))]
    frames[-1].save(BUILD / "browser-demo.png")                      # the finished recommendation


def markdown():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    head, sep, rest = text.partition("\nTACC AgentForge is the course project")
    if not sep:
        raise SystemExit("README header changed; update readme_to_tex.py")
    tagline = re.search(r"<strong>(.*?)</strong>", head).group(1)
    rest = sep.lstrip("\n") + rest
    gif = re.search(r'<p align="center">\s*<img src="docs/images/browser-demo.gif"[^>]*alt="([^"]*)"[^>]*>\s*</p>', rest)
    rest = rest.replace(gif.group(0), f"![{gif.group(1)}](build/browser-demo.png)")
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
    # Code blocks: wrap long lines instead of running off the page.
    tex = tex.replace("\\begin{verbatim}", "\\begin{Verbatim}[breaklines,breakanywhere,fontsize=\\footnotesize,frame=lines,framesep=2pt]")
    tex = tex.replace("\\end{verbatim}", "\\end{Verbatim}")
    # Images: fixed width, no alt key.
    tex = re.sub(r"\\pandocbounded\{\\includegraphics\[[^\]]*\]\{([^}]*)\}\}",
                 lambda m: "\\begin{center}\\includegraphics[width=0.56\\linewidth]{" + m.group(1) + "}\\end{center}", tex)
    # Code font: allow line breaks after "/" and "_" in paths and file names.
    tex = re.sub(r"\\texttt\{((?:\\[{}]|[^{}])*)\}",
                 lambda m: "\\texttt{" + re.sub(r"(?<!\\)/", "/\\\\allowbreak{}", m.group(1)).replace("\\_", "\\_\\allowbreak{}") + "}", tex)
    # Tables: longtable -> tabular, so no table breaks across pages; keep booktabs rules.
    def table(m):
        spec, body = m.group(1), m.group(2)
        if "p{" not in spec:
            # pandoc left short-cell tables as l/r/c columns, which do not wrap: give every column a width
            # proportional to its longest cell (at least 8%), so the table fits the text width.
            cols = len(re.findall(r"[lrc]", spec.replace("@{}", "")))
            rows = [r for r in re.split(r"\\\\", body) if "&" in r]
            longest, word = [1] * cols, [1] * cols
            for row in rows:
                for i, cell in enumerate(row.split("&")[:cols]):
                    plain = re.sub(r"\\[a-zA-Z]+|[{}]", "", cell).strip()
                    longest[i] = max(longest[i], len(plain))
                    word[i] = max([word[i]] + [len(w) for w in re.split(r"[\s/]+", plain)])
            # At least the longest unbreakable word (about 105 footnotesize characters fit the text width).
            floor = [max(0.08, (w + 2) / 105) for w in word]
            share = [max(f, n / sum(longest)) for f, n in zip(floor, longest)]
            share = [s / sum(share) for s in share]
            spec = "@{}" + "".join(
                f">{{\\raggedright\\arraybackslash}}p{{(\\linewidth - {2 * cols}\\tabcolsep) * \\real{{{s:.4f}}}}}"
                for s in share) + "@{}"
        lines = [l for l in body.split("\n") if l.strip() not in ("\\endhead", "\\endfirsthead", "\\endlastfoot",
                                                                     "\\endfoot", "\\bottomrule\\noalign{}")]
        return ("\\par\\smallskip\\noindent{\\footnotesize\\begin{tabular}{" + spec + "}\n" + "\n".join(lines).rstrip()
                + "\n\\bottomrule\n\\end{tabular}}\\par\\smallskip\n")
    tex = re.sub(r"\\begin\{longtable\}\[\]\{(.*?)\}\n(.*?)\\end\{longtable\}", table, tex, flags=re.S)
    return tex


def main():
    BUILD.mkdir(exist_ok=True)
    gif_frame()
    (BUILD / "readme.tex").write_text(to_latex(markdown()), encoding="utf-8")
    print("wrote", BUILD / "readme.tex")


if __name__ == "__main__":
    main()
