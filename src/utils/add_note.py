"""Utility function to append notes to a markdown file."""

import os
from typing import Union, Optional, List
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def add_note(
    title: str,
    notes_path = None,
    config_path: Optional[str] = None,
    content: Union[str, list, dict, None, Path] = None,
    text: Optional[str] = None,
) -> None:
    """Append a note section to a markdown file.

    Args:
        notes_path: Path to the notes.md file. Created if it doesn't exist.
        title: Title for the note section (rendered as ## header).
        content: The main content to add. Can be:
            - str ending with image extension (.png, .jpg, .jpeg, .gif, .svg, .bmp, .webp):
                rendered as a markdown image.
            - str ending with .txt or other text file extension:
                contents of the file are read and inserted.
            - str (plain text): inserted as-is.
            - dict: rendered as a markdown table (keys = columns).
            - list of dicts: rendered as a markdown table (keys = columns, rows = items).
            - list of lists: rendered as a markdown table (first row = header).
            - list of str: rendered as a markdown table with a single column.
            - None: no content added (only title and optional text).
        text: Optional additional text to add below the title.
    """
    global logger
    sections: List[str] = []
    # Check if a logger is configured, if not, use simple logger
    if not logger.hasHandlers():
        # setup basic logger to print to console
        logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
        logger = logging.getLogger(__name__)
        
    if notes_path is None and config_path is not None:
        # Open config as json
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        # Try to get output_dir from config
        output_dir = config["logging"]["output_dir"]
        notes_path = Path(output_dir) / "notes.md"
        # Check if output_dir exists
        if not Path(output_dir).exists():
            logger.warning(f"Output directory from config does not exist: {output_dir}.")
    elif notes_path is None and config_path is None:
        logging.warning("No notes_path or config_path provided. Skipping note addition.")
        # Attempted notes path
    
    # Check if notes_path exists
    if notes_path is not None:
        notes_path = Path(notes_path)
        if not notes_path.exists():
            logger.debug(f"Creating new notes file: {notes_path}")
            notes_path.parent.mkdir(parents=True, exist_ok=True)    
        
    # Title
    sections.append(f"## {title}\n")

    # Optional text
    if text is not None:
        sections.append(f"{text}\n")

    # Notes path
    notes_path = str(notes_path)  # Ensure it's a string path, not a Path object
    # Content
    if content is not None:
        rendered = _render_content(content, notes_path=notes_path)
        if rendered:
            sections.append(rendered)

    block = "\n".join(sections) + "\n"
    # Check if block already exists in the file to avoid duplicates
    if os.path.exists(notes_path):
        with open(notes_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
            if block.strip() in existing_content:
                logger.debug("Note already exists in the file. Skipping append. Notes path: %s", notes_path)
                return

    # Append or create
    mode = "a" if os.path.exists(notes_path) else "w"
    with open(notes_path, mode, encoding="utf-8") as f:
        # Add a blank line separator if appending to an existing non-empty file
        if mode == "a":
            f.write("\n")
        f.write(block)
    logger.info(f"Note added to {notes_path} successfully.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".webp"}
_TEXT_EXTENSIONS = {".txt", ".log", ".csv", ".tsv", ".md", ".rst"}


def _render_content(content: Union[str, list, dict, Path], notes_path) -> str:
    """Return a markdown string for the given content."""
    if isinstance(content, str):
        return _render_string(content, notes_path)
    if isinstance(content, Path):
        return _render_string(str(content), notes_path)
    if isinstance(content, dict):
        return _dict_to_table(content)
    if isinstance(content, list):
        return _list_to_table(content)
    return str(content)


def _render_string(s: str, notes_path = None) -> str:
    ext = os.path.splitext(s)[1].lower()
    logger.debug(f"Rendering string content: {s} (ext: {ext})")
    if ext in _IMAGE_EXTENSIONS:
        # If the image path is absolute, try to make it relative to the notes file for better portability
        if os.path.isabs(s) and notes_path is not None:
            s = os.path.relpath(s, os.path.dirname(notes_path))
        label = os.path.basename(s)
        return f"![{label}]({s})\n"
    if ext in _TEXT_EXTENSIONS and os.path.isfile(s):
        with open(s, "r", encoding="utf-8") as f:
            return f.read().rstrip("\n") + "\n"
    return s + "\n"


def _dict_to_table(d: dict) -> str:
    """Render a single dict as a two-column (Key | Value) table."""
    lines = ["| Key | Value |", "| --- | --- |"]
    for k, v in d.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def _list_to_table(lst: list) -> str:
    """Render a list as a markdown table.

    - list[dict]: each dict is a row, keys are column headers.
    - list[list]: first sub-list is the header row.
    - list[str/scalar]: single-column table.
    """
    if not lst:
        return ""

    # list of dicts
    if isinstance(lst[0], dict):
        headers = list(lst[0].keys())
        lines = [
            "| " + " | ".join(str(h) for h in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in lst:
            lines.append(
                "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
            )
        return "\n".join(lines) + "\n"

    # list of lists
    if isinstance(lst[0], (list, tuple)):
        header = lst[0]
        lines = [
            "| " + " | ".join(str(h) for h in header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in lst[1:]:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines) + "\n"

    # list of scalars → single column
    lines = ["| Value |", "| --- |"]
    for item in lst:
        lines.append(f"| {item} |")
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    # Example usage
    add_note(
        config_path="/home/ap/cloud/Master/aris_master/queue/scheduled/Aresolution_128.json",
        title="Test Note",
        content=[
            {"Metric": "Accuracy", "Value": "95%"},
            {"Metric": "Loss", "Value": "0.05"},
        ],
        text="This is a test note with a table of metrics.",
    )