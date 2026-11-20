#!/usr/bin/env bash
# Build report/main.pdf: the project report with the repository README as its appendix.
# Needs a TeX Live with newtx, pgfplots and fvextra, pandoc, and Python 3 with Pillow.
set -euo pipefail
cd "$(dirname "$0")"
(cd figures && pdflatex -interaction=nonstopmode -halt-on-error architecture-standalone.tex > /dev/null)
python3 readme_to_tex.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -silent main.tex
echo "built $(pwd)/main.pdf"
