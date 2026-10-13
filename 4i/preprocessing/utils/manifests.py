"""Atomic manifest, hashing, and selection-table helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import math


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: str | Path, values) -> None:
    atomic_write_text(
        path,
        json.dumps(json_safe(values), indent=2, sort_keys=True, allow_nan=False),
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return json_safe(value.item())
    except ImportError:
        pass
    return value


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    if not rows:
        atomic_write_text(path, "")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            fieldnames = list(rows[0])
            for row in rows[1:]:
                fieldnames.extend(key for key in row if key not in fieldnames)
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def sha256_json(value) -> str:
    """Hash a JSON-compatible value with deterministic ordering."""
    encoded = json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def approved_selections(path: str | Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    selections = {row.get("acquisition_id", ""): row for row in rows}
    missing_ids = [index for index, row in enumerate(rows, 2) if not row.get("acquisition_id")]
    if missing_ids or len(selections) != len(rows):
        raise ValueError("Selection file has blank or duplicate acquisition IDs.")
    for acquisition_id, row in selections.items():
        if row.get("review_status", "").strip().lower() != "approved":
            raise ValueError(f"{acquisition_id} has not been approved.")
        if row.get("chosen_method") not in {"BaSiC", "Harmony-like"}:
            raise ValueError(
                f"{acquisition_id} chosen_method must be BaSiC or Harmony-like."
            )
    return selections


def combine_csv_files(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows = []
    for path in sorted(paths):
        rows.extend(read_csv(path))
    return rows
