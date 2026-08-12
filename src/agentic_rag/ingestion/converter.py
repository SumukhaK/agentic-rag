from pathlib import Path

from markitdown import MarkItDown

_converter = MarkItDown()


def convert_to_markdown(path: Path) -> str:
    """Convert any supported source file to Markdown text via markitdown."""
    result = _converter.convert(str(path))
    return result.text_content
