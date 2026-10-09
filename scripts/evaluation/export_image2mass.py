"""Export locally obtained legacy Image2Mass data without executing its code.

Run with a Python runtime that already has h5py and lz4. Nothing is downloaded,
installed, unpickled, or deserialized as executable Keras/Python code here.
The original files and exports belong in ignored outputs, not in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickletools

import numpy as np


def read_density_values(path: Path) -> np.ndarray:
    """Accept only the released pickle's single list of numeric constants."""
    import lz4.block

    return density_constants(lz4.block.decompress(path.read_bytes()))


def density_constants(raw: bytes) -> np.ndarray:
    """Parse the constrained numeric-list format without unpickling anything."""
    operations = list(pickletools.genops(raw))
    names = [op.name for op, _, _ in operations]
    expected = ["PROTO", "FRAME", "EMPTY_LIST", "MEMOIZE", "MARK"]
    if names[:5] != expected or names[-2:] != ["APPENDS", "STOP"]:
        raise ValueError("unexpected density-table serialization")
    if any(name != "BINFLOAT" for name in names[5:-2]):
        raise ValueError("density table must contain only numeric constants")
    values = np.asarray([value for _, value, _ in operations[5:-2]], np.float32)
    if values.size < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("invalid density values")
    if (np.diff(values) < 0).any():
        raise ValueError("density table must be ordered")
    return values


def export_release(source: Path, destination: Path) -> None:
    import h5py

    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("models/shape_aware.h5", "thickness_model.h5"):
        path = source / name
        expected = {
            "models/shape_aware.h5": "a49f740b7de8bc27f7e88a12deb6862f504dc44bdf851c50987f48e9278b82a3",
            "thickness_model.h5": "1b4ea1ac10503f227ff9cea57345fc8c6f94c3eff8ecc0e38f2de5f6f2910b79",
        }
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise ValueError(f"not the audited released checkpoint: {name}")
        arrays = {}
        with h5py.File(path, "r") as handle:
            configuration = handle.attrs["model_config"]
            if isinstance(configuration, bytes):
                configuration = configuration.decode("utf-8")
            # JSON stays data. In particular, serialized Lambda functions are
            # never passed to marshal, eval, exec, Keras, or pickle.loads.
            config = json.loads(configuration)
            handle["model_weights"].visititems(
                lambda key, item: (
                    arrays.__setitem__(key, np.asarray(item))
                    if isinstance(item, h5py.Dataset)
                    else None
                )
            )
            keras_version = str(handle.attrs.get("keras_version"))
        stem = path.stem
        (destination / f"{stem}.json").write_text(json.dumps(config), encoding="utf-8")
        np.savez(destination / f"{stem}.npz", **arrays)
        records.append(
            {
                "file": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "keras_version": keras_version,
                "numeric_arrays": len(arrays),
            }
        )
    density_path = source / "score_to_percentile.pklz"
    if (
        hashlib.sha256(density_path.read_bytes()).hexdigest()
        != "f7bb0de68ca1860479dbc83e15ede4ebeb272c2562b8214fa57de5d9f62b3e74"
    ):
        raise ValueError("not the audited released density table")
    density = read_density_values(density_path)
    np.save(destination / "density.npy", density, allow_pickle=False)
    records.append(
        {
            "file": density_path.name,
            "sha256": hashlib.sha256(density_path.read_bytes()).hexdigest(),
            "values": len(density),
        }
    )
    (destination / "export.json").write_text(json.dumps(records, indent=2), "utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_release(args.source, args.output)
