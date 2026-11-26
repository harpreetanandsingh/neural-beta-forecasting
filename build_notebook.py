"""Assemble cells/step*.py into neural_beta.ipynb, execute it, export HTML.

Cell syntax inside each step file:
    # %%              -> starts a code cell
    # %% [markdown]   -> starts a markdown cell (lines are un-commented: leading '# ' stripped)

Usage:  python build_notebook.py            # all steps, execute, export html
        python build_notebook.py --upto 3   # only steps 0..3 (for development)
        python build_notebook.py --no-exec
"""
import argparse
import glob
import os
import re
import subprocess

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "neural_beta.ipynb")


def parse(path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "md":
            text = "\n".join(re.sub(r"^# ?", "", ln) for ln in text.splitlines())
            cells.append(new_markdown_cell(text))
        else:
            cells.append(new_code_cell(text))

    for line in open(path).read().splitlines():
        if line.startswith("# %% [markdown]"):
            flush(); kind, buf = "md", []
        elif line.startswith("# %%"):
            flush(); kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=99)
    ap.add_argument("--no-exec", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(HERE, "cells", "step*.py")))
    files = [f for f in files if int(re.search(r"step(\d+)", f).group(1)) <= a.upto]
    nb = new_notebook()
    for f in files:
        nb.cells.extend(parse(f))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.write(nb, NB)
    print(f"wrote {NB} with {len(nb.cells)} cells from {len(files)} files")
    if a.no_exec:
        return
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=-1", NB], check=True, cwd=HERE)
    subprocess.run(["jupyter", "nbconvert", "--to", "html", NB], check=True, cwd=HERE)


if __name__ == "__main__":
    main()
