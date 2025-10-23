"""
Generate a student version of Jupyter notebooks by replacing
cells tagged as 'solution' with empty code cells containing a placeholder.

Usage:
    python devtools/make_student.py input_notebook.ipynb output_notebook.ipynb
"""

from __future__ import annotations
import sys
from pathlib import Path
import nbformat
from nbformat.notebooknode import NotebookNode


def make_student_version(
    input_path: str | Path,
    output_path: str | Path,
    tag_to_replace: str = "solution",
) -> None:
    """
    Create a student version of a Jupyter notebook by replacing all cells
    tagged with `tag_to_replace` by placeholder code cells.

    Parameters
    ----------
    input_path : str | Path
        Path to the input notebook (typically the solutions version).
    output_path : str | Path
        Path where the cleaned student notebook should be written.
    tag_to_replace : str, optional
        Tag identifying cells that should be replaced. Default is 'solution'.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    nb: NotebookNode = nbformat.read(input_path, as_version=4)
    new_cells: list[NotebookNode] = []

    for cell in nb.cells:
        tags: list[str] = cell.get("metadata", {}).get("tags", [])

        if tag_to_replace in tags:
            # Replace tagged cell with placeholder
            placeholder = nbformat.v4.new_code_cell(
                source="# your solution here",
                metadata={"tags": ["placeholder"]},
            )
            new_cells.append(placeholder)
        else:
            # Clean up execution metadata
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
            new_cells.append(cell)

    nb.cells = new_cells
    nbformat.write(nb, output_path)
    print(f"✅ Wrote student notebook: {output_path}")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) != 3:
        print("Usage: python devtools/make_student.py input.ipynb output.ipynb")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    make_student_version(input_path, output_path)


if __name__ == "__main__":
    main()
