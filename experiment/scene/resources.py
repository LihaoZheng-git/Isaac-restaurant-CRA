"""Locate the restaurant resources selected by the experiment launcher."""

import json
import os
from pathlib import Path


DATA = Path(__file__).resolve().with_name("data")


def live_assets():
    """Resolve local converted props and characters without opening an Isaac stage."""
    selected = os.environ.get("CRA_SCENE_ROOT")
    scene_root = Path(selected) if selected else DATA.parent / "prepared"
    locations = scene_root / "asset-locations.json"
    if selected or locations.is_file():
        return Path(json.loads(locations.read_text(encoding="utf-8"))["live"])
    # The native module also supports importing scene constants outside a run.
    return scene_root / "scenes" / "generated"
