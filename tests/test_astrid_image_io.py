import zipfile
from io import BytesIO

import pytest
from PIL import Image

from astrid_image_io import (
    ArchiveSafetyError,
    ArchiveSafetyLimits,
    open_image_with_pixel_limit,
    screen_zip_members,
)


def _zip_bytes(files, compression=zipfile.ZIP_DEFLATED):
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return output.getvalue()


def _png_bytes(size=(8, 8)):
    output = BytesIO()
    Image.new("RGB", size, color=(20, 40, 60)).save(output, format="PNG")
    return output.getvalue()


def test_screen_zip_members_accepts_small_archive():
    archive, members, warnings = screen_zip_members(
        _zip_bytes({"class_a/sample.png": _png_bytes()}),
        ArchiveSafetyLimits(),
    )
    try:
        assert [member.filename for member in members] == ["class_a/sample.png"]
        assert warnings == []
        assert archive.read(members[0]).startswith(b"\x89PNG")
    finally:
        archive.close()


def test_screen_zip_members_rejects_large_total_expansion():
    data = _zip_bytes({"large.txt": b"x" * 2048})

    with pytest.raises(ArchiveSafetyError, match="expands to"):
        screen_zip_members(
            data,
            ArchiveSafetyLimits(max_total_uncompressed_bytes=1024),
        )


def test_screen_zip_members_rejects_large_archive_upload():
    data = _zip_bytes({"sample.txt": b"small"}, compression=zipfile.ZIP_STORED)

    with pytest.raises(ArchiveSafetyError, match="ZIP is"):
        screen_zip_members(data, ArchiveSafetyLimits(max_archive_bytes=len(data) - 1))


def test_screen_zip_members_rejects_too_many_members():
    data = _zip_bytes({"one.txt": b"1", "two.txt": b"2"})

    with pytest.raises(ArchiveSafetyError, match="contains 2 files"):
        screen_zip_members(data, ArchiveSafetyLimits(max_members=1))


def test_screen_zip_members_skips_oversized_member():
    data = _zip_bytes({"large.png": b"x" * 2048}, compression=zipfile.ZIP_STORED)
    archive, members, warnings = screen_zip_members(
        data,
        ArchiveSafetyLimits(
            max_total_uncompressed_bytes=4096,
            max_member_uncompressed_bytes=1024,
        ),
    )
    try:
        assert members == []
        assert "per-file limit" in warnings[0]
    finally:
        archive.close()


def test_screen_zip_members_skips_suspicious_compression_ratio():
    data = _zip_bytes({"repetitive.png": b"0" * (1024 * 1024)})
    archive, members, warnings = screen_zip_members(
        data,
        ArchiveSafetyLimits(
            max_total_uncompressed_bytes=2 * 1024 * 1024,
            max_member_uncompressed_bytes=2 * 1024 * 1024,
            max_compression_ratio=10,
        ),
    )
    try:
        assert members == []
        assert "compression ratio" in warnings[0]
    finally:
        archive.close()


def test_screen_zip_members_rejects_invalid_zip():
    with pytest.raises(ArchiveSafetyError, match="Invalid ZIP"):
        screen_zip_members(b"not a zip", ArchiveSafetyLimits())


def test_open_image_with_pixel_limit_rejects_large_dimensions():
    with pytest.raises(ValueError, match="400 pixels"):
        open_image_with_pixel_limit(_png_bytes((20, 20)), max_pixels=399)


def test_open_image_with_pixel_limit_loads_image_within_limit():
    image = open_image_with_pixel_limit(_png_bytes((20, 20)), max_pixels=400)
    try:
        assert image.size == (20, 20)
    finally:
        image.close()
