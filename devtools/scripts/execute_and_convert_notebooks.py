#!/usr/bin/env python3
"""Execute and convert notebooks while skipping cells that have a given tag.

This script:
 - finds .ipynb files under --input-dir
 - loads each notebook, removes cells that have the skip tag
 - executes the notebook with nbconvert ExecutePreprocessor
 - exports the executed notebook to a markdown file under --output-dir

Use this in CI to skip interactive or long-running cells by adding a tag
to those cells, e.g. tags: ["ci_skip"]

Example:
  python devtools/scripts/execute_and_convert_notebooks.py \
    --input-dir notebooks_with_solutions --output-dir notebooks-rendered \
    --skip-tag ci_skip --timeout 600
"""
import argparse
import nbformat
from nbformat import NotebookNode
import sys
from pathlib import Path
from typing import List, Optional
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.exporters import MarkdownExporter
from copy import deepcopy


def notebook_files(input_dir: str) -> List[Path]:
    p = Path(input_dir)
    return sorted(p.rglob("*.ipynb"))


def remove_tagged_cells(nb: NotebookNode, tag: Optional[str]) -> List[NotebookNode]:
    if not tag:
        return list(nb.cells)
    return [
        cell for cell in nb.cells if tag not in cell.get("metadata", {}).get("tags", [])
    ]


def has_skip_tag(cell: NotebookNode, tag: Optional[str]) -> bool:
    if not tag:
        return False
    return tag in cell.get("metadata", {}).get("tags", [])


def execute_notebook(
    nb: NotebookNode,
    timeout: int,
    kernel_name: str,
    cwd: Optional[str] = None,
    skip_tag: Optional[str] = None,
) -> NotebookNode:
    """Execute the notebook but skip execution of cells that have skip_tag.

    Implementation: run a deep copy of the notebook where skipped code cells
    are replaced with a noop (`pass`) so the ExecutePreprocessor executes but
    does nothing for those cells. After execution, copy outputs and
    execution_count back to the original notebook so the original cell
    sources are preserved for conversion.
    """
    exec_nb = deepcopy(nb)
    # replace skipped code cells with a harmless noop so they won't run
    for cell in exec_nb.cells:
        if cell.get("cell_type") == "code" and has_skip_tag(cell, skip_tag):
            cell.source = "pass\n"
            # clear any existing outputs
            cell.outputs = []
            cell.execution_count = None

    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)
    ep.preprocess(exec_nb, {"metadata": {"path": cwd or "."}})

    # copy outputs back to original notebook cells
    for orig_cell, run_cell in zip(nb.cells, exec_nb.cells):
        if orig_cell.get("cell_type") == "code":
            orig_cell["outputs"] = run_cell.get("outputs", [])
            orig_cell["execution_count"] = run_cell.get("execution_count", None)

    return nb


def convert_to_markdown(nb, out_path: Path):
    exporter = MarkdownExporter()
    body, resources = exporter.from_notebook_node(nb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf8")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--skip-tag", default="ci_skip", help="Cell tag to remove before execution"
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="ExecutePreprocessor timeout in seconds",
    )
    p.add_argument(
        "--kernel", default="python3", help="Kernel name to use for execution"
    )
    args = p.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    files = notebook_files(input_dir)
    if not files:
        print(f"No notebooks found under {input_dir}")
        return 0

    exit_code = 0
    for nb_path in files:
        rel = nb_path.relative_to(input_dir)
        out_md = output_dir / rel.with_suffix(".md")
        print(f"Processing {nb_path} -> {out_md}")
        try:
            nb = nbformat.read(str(nb_path), as_version=4)
            # execute in the notebook's parent directory to keep relative paths working
            cwd = str(nb_path.parent)
            nb = execute_notebook(
                nb,
                timeout=args.timeout,
                kernel_name=args.kernel,
                cwd=cwd,
                skip_tag=args.skip_tag,
            )
            convert_to_markdown(nb, out_md)
        except Exception as e:
            print(f"ERROR processing {nb_path}: {e}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
