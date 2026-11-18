"""Guard the three phase commands used in the recorded v2 mission (83abdbe)."""

import hashlib
from pathlib import Path

from experiment.mission import ROOT, read_recipe


def test_relocated_mission_uses_recorded_v2_commands(tmp_path):
    commands, _ = read_recipe(ROOT / "experiment/configs/mission.json", tmp_path)
    assert list(commands) == ["left", "return", "right"]
    # Hash each archived argument vector with machine-specific prefixes removed.
    # This covers the controller options, encounter selectors and their ordering.
    expected = {
        "left": "40525866f2615044be709f39a31639bf0359bf67f1a91213c91e9d2d862bdaee",
        "return": "0479199afd9faf3b1bc812a65beedd724048c7d68d907b2ef552dec59622dbdb",
        "right": "d29f8c500a6ba19bf3bef0b010b0cf713151519fa64b325e7b4a79f43343fea3",
    }
    for phase, command in commands.items():
        assert (ROOT / command[0]).is_file()
        options = command[1:]
        for flag in (
            "--restaurant-scene", "--restaurant-layout-survey",
            "--left-outbound-cast-manifest", "--right-crossing-cast-manifest",
        ):
            if flag in options:
                index = options.index(flag) + 1
                asset = Path(options[index])
                assert asset.is_relative_to(tmp_path)
                options[index] = asset.name
        digest = hashlib.sha256("\n".join(options).encode("utf-8")).hexdigest()
        assert digest == expected[phase], phase
