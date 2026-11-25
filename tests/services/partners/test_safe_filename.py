"""Regression tests for partners.helpers.safe_filename."""

from __future__ import annotations

from pathlib import Path

from deeptutor.partners.helpers import safe_filename


def test_safe_filename_strips_embedded_null_byte(tmp_path: Path) -> None:
    cleaned = safe_filename("photo\x00.png")
    assert "\x00" not in cleaned
    assert cleaned.endswith(".png")
    # The point is that the cleaned name is usable on the local filesystem;
    # where that filesystem lives is the platform's business (a hardcoded
    # `/tmp` resolves to the drive root on Windows).
    path = tmp_path / f"abc_{cleaned}"
    path.write_bytes(b"x")
    assert path.read_bytes() == b"x"
    path.unlink()


def test_safe_filename_strips_newlines_and_dot_names() -> None:
    assert "\n" not in safe_filename("a\nb.png")
    assert safe_filename(".") == ""
    assert safe_filename("..") == ""
