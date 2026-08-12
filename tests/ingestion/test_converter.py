import pytest

from agentic_rag.ingestion.converter import convert_to_markdown


def test_convert_to_markdown_returns_text_content_for_txt_file(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("Manchester United won 2-1.")

    markdown = convert_to_markdown(source)

    assert "Manchester United won 2-1." in markdown


def test_convert_to_markdown_raises_for_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.txt"

    with pytest.raises(Exception):
        convert_to_markdown(missing)
