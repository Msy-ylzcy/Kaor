"""Estimate ZIP deflate size for release inputs without creating an archive."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path


def iter_files(paths: list[Path], pattern: str) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.match(pattern):
            files.add(path.resolve())
        elif path.is_dir():
            files.update(candidate.resolve() for candidate in path.rglob(pattern) if candidate.is_file())
    return sorted(files)


def deflated_size(path: Path) -> int:
    compressor = zlib.compressobj(level=6, method=zlib.DEFLATED, wbits=-15)
    total = 0
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            total += len(compressor.compress(block))
    return total + len(compressor.flush())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--pattern", default="*")
    args = parser.parse_args()
    files = iter_files(args.paths, args.pattern)
    raw = sum(path.stat().st_size for path in files)
    compressed = sum(deflated_size(path) for path in files)
    print(
        json.dumps(
            {
                "files": len(files),
                "raw_bytes": raw,
                "deflated_bytes": compressed,
                "ratio": compressed / raw if raw else 0.0,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
