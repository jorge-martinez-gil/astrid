"""Resource-safe helpers for image archive ingestion."""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

MIB = 1024 * 1024


class ArchiveSafetyError(ValueError):
    """Raised when an archive exceeds a global ingestion safety limit."""


@dataclass(frozen=True)
class ArchiveSafetyLimits:
    max_archive_bytes: int = 512 * MIB
    max_members: int = 20_000
    max_total_uncompressed_bytes: int = 2 * 1024 * MIB
    max_member_uncompressed_bytes: int = 100 * MIB
    max_compression_ratio: float = 200.0
    max_image_pixels: int = 50_000_000


def screen_zip_members(
    zip_bytes: bytes,
    limits: ArchiveSafetyLimits,
) -> tuple[zipfile.ZipFile, list[zipfile.ZipInfo], list[str]]:
    """Validate global ZIP limits and remove unsafe individual members."""

    if len(zip_bytes) > limits.max_archive_bytes:
        raise ArchiveSafetyError(
            f"ZIP is {len(zip_bytes) / MIB:.1f} MiB; limit is "
            f"{limits.max_archive_bytes / MIB:.1f} MiB."
        )

    try:
        archive = zipfile.ZipFile(BytesIO(zip_bytes))
        members = [info for info in archive.infolist() if not info.is_dir()]
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArchiveSafetyError(f"Invalid ZIP archive: {exc}") from exc

    if len(members) > limits.max_members:
        archive.close()
        raise ArchiveSafetyError(
            f"ZIP contains {len(members):,} files; limit is {limits.max_members:,}."
        )

    total_uncompressed = sum(info.file_size for info in members)
    if total_uncompressed > limits.max_total_uncompressed_bytes:
        archive.close()
        raise ArchiveSafetyError(
            f"ZIP expands to {total_uncompressed / MIB:.1f} MiB; limit is "
            f"{limits.max_total_uncompressed_bytes / MIB:.1f} MiB."
        )

    accepted: list[zipfile.ZipInfo] = []
    oversized = 0
    suspicious_ratio = 0
    for info in members:
        if info.file_size > limits.max_member_uncompressed_bytes:
            oversized += 1
            continue

        ratio = info.file_size / max(1, info.compress_size)
        if info.file_size >= MIB and ratio > limits.max_compression_ratio:
            suspicious_ratio += 1
            continue
        accepted.append(info)

    warnings: list[str] = []
    if oversized:
        warnings.append(
            f"Skipped {oversized:,} file(s) larger than the per-file limit of "
            f"{limits.max_member_uncompressed_bytes / MIB:.1f} MiB."
        )
    if suspicious_ratio:
        warnings.append(
            f"Skipped {suspicious_ratio:,} file(s) with compression ratio above "
            f"{limits.max_compression_ratio:.0f}:1."
        )
    return archive, accepted, warnings


def open_image_with_pixel_limit(data: bytes, max_pixels: int) -> Image.Image:
    """Decode an image only when its declared dimensions fit the pixel budget."""

    image = Image.open(BytesIO(data))
    width, height = image.size
    pixels = width * height
    if pixels > max_pixels:
        image.close()
        raise ValueError(f"image has {pixels:,} pixels; limit is {max_pixels:,} pixels")
    image.load()
    return image
