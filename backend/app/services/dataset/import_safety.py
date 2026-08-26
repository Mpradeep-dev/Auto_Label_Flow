"""Shared hardening for the COCO/CVAT zip importers (audit finding SEC-03).

Both `import_coco.py` and `import_cvat.py` extract a user-uploaded zip and
(for CVAT) parse an XML file out of it. Neither had any protection against
a hostile archive before this module existed:

  - a zip whose entries resolve outside the extraction directory
    ("Zip Slip" path traversal)
  - a zip that decompresses to far more data than its compressed size
    suggests (a "zip bomb"), exhausting disk on the same host running
    Postgres and local file storage
  - an XML file with a DOCTYPE declaring custom entities, letting a tiny
    file expand to gigabytes in memory ("billion laughs") or attempt to
    read local files via an external entity (XXE)

No new dependency is introduced — `defusedxml` isn't in requirements.txt,
and the fix here (reject any DOCTYPE outright) covers the same attack
classes a CVAT export never legitimately needs.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import xml.parsers.expat
import zipfile
from pathlib import Path

# A legitimate dataset export (images + annotation JSON/XML) never needs to
# be this large relative to a 200MB upload cap, and it never compresses
# anywhere near this hard — both are generous enough to never trip on real
# COCO/CVAT exports while still bounding worst-case disk/memory use.
_MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB
_MAX_COMPRESSION_RATIO = 200


class UnsafeArchiveError(RuntimeError):
    pass


def safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    """Validate every member of `zf` before extracting any of them, then
    extract into `dest`. Raises UnsafeArchiveError instead of writing
    anything if a member would escape `dest` or the archive's total
    decompressed size / any single entry's compression ratio looks like a
    bomb rather than a real dataset export."""
    dest = dest.resolve()
    total_uncompressed = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        member_path = (dest / info.filename).resolve()
        try:
            member_path.relative_to(dest)
        except ValueError:
            raise UnsafeArchiveError(f"Archive entry escapes the extraction directory: {info.filename!r}") from None

        total_uncompressed += info.file_size
        if total_uncompressed > _MAX_UNCOMPRESSED_BYTES:
            raise UnsafeArchiveError("Archive would decompress to more than the allowed size limit")
        if info.compress_size > 0 and info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO:
            raise UnsafeArchiveError(f"Archive entry {info.filename!r} has a suspiciously high compression ratio")

    zf.extractall(dest)


class UnsafeXmlError(RuntimeError):
    pass


def parse_xml_safely(path: Path) -> ET.Element:
    """Parse `path`, rejecting any DOCTYPE/entity declaration outright. A
    real CVAT-XML export never declares one, so this costs nothing on
    legitimate input while blocking both entity-expansion bombs and XXE at
    the same time (no external-entity allow-list to get wrong).

    Two passes, deliberately: `ElementTree.XMLParser` no longer exposes its
    underlying expat parser as a public attribute (that shape has changed
    across Python versions), so the first pass uses the stable, public
    `xml.parsers.expat` API purely to detect a DOCTYPE and raise before any
    entity has a chance to expand. Only once that pass confirms there is no
    DOCTYPE does the second pass build the real tree with plain `ET.parse`."""
    scanner = xml.parsers.expat.ParserCreate()

    def _reject_doctype(*_args: object) -> None:
        raise UnsafeXmlError("XML documents with a DOCTYPE/entity declaration are not allowed")

    scanner.StartDoctypeDeclHandler = _reject_doctype
    with open(path, "rb") as f:
        scanner.ParseFile(f)

    return ET.parse(path).getroot()
