"""Isolated-hydrogen guard (reject_isolated_hydrogens).

A hydrogen whose X-H bond is stretched past the connectivity cutoff used to come
back as a free ``[H+]`` balanced by a carbanion/alkoxide, rated high confidence
under a declared total charge. See piskuliche/xyzrecover#1.
"""

import math

import pytest

from xyzrecover import PerceptionConfig, recover_xyz_text
from xyzrecover.cli import main


def methanol(d_oh: float) -> str:
    """Methanol with the hydroxyl O-H bond length set to ``d_oh`` angstrom."""
    angle = math.radians(70.0)
    hx, hy = 1.43 + d_oh * math.cos(angle), d_oh * math.sin(angle)
    return f"""6
methanol O-H={d_oh}
C 0.0000 0.0000 0.0000
O 1.4300 0.0000 0.0000
H -0.3600 1.0300 0.0000
H -0.3600 -0.5150 0.8920
H -0.3600 -0.5150 -0.8920
H {hx:.4f} {hy:.4f} 0.0000
"""


H2 = """2
dihydrogen
H 0 0 0
H 0 0 0.74
"""

H_ATOM = """1
hydrogen atom
H 0 0 0
"""


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("total_charge", [0, None])
def test_stretched_oh_is_rejected(split, total_charge):
    config = PerceptionConfig(total_charge=total_charge, split_fragments=split)
    recs = recover_xyz_text(methanol(1.8), config)
    assert recs
    for rec in recs:
        assert rec.status == "failed"
        assert rec.smiles is None
        assert rec.inchikey is None
        assert rec.confidence == "none"
    assert any("[5]" in err for rec in recs for err in rec.errors)  # names atom index 5
    # Every atom is still accounted for across the failed records.
    assert sorted(i for rec in recs for i in rec.atom_indices) == list(range(6))


@pytest.mark.parametrize("split", [False, True])
def test_equilibrium_geometry_unchanged(split):
    rec = recover_xyz_text(methanol(0.96), PerceptionConfig(total_charge=0, split_fragments=split))
    assert len(rec) == 1
    assert rec[0].status == "ok"
    assert rec[0].smiles == "CO"
    assert rec[0].confidence == "high"


def test_dihydrogen_is_accepted():
    rec = recover_xyz_text(H2, PerceptionConfig(total_charge=0))[0]
    assert rec.status == "ok"
    assert rec.formula == "H2"
    assert rec.smiles == "[H][H]"


def test_single_hydrogen_block_is_not_rejected():
    recs = recover_xyz_text(H_ATOM)
    assert len(recs) == 1
    assert not any("no bonded neighbour" in err for err in recs[0].errors)


@pytest.mark.parametrize("split", [False, True])
def test_option_off_restores_previous_behaviour(split):
    config = PerceptionConfig(
        total_charge=0, split_fragments=split, reject_isolated_hydrogens=False
    )
    recs = recover_xyz_text(methanol(1.8), config)
    assert all(rec.status == "ok" for rec in recs)
    smiles = ".".join(rec.smiles for rec in recs)
    assert "[O-]" in smiles


def test_cli_flag_disables_guard(tmp_path):
    path = tmp_path / "stretched.xyz"
    path.write_text(methanol(1.8))
    out_default = tmp_path / "default.json"
    out_allowed = tmp_path / "allowed.json"
    main([str(path), "--total-charge", "0", "--no-split", "--json", str(out_default)])
    main(
        [
            str(path),
            "--total-charge",
            "0",
            "--no-split",
            "--allow-isolated-hydrogens",
            "--json",
            str(out_allowed),
        ]
    )
    assert '"status": "failed"' in out_default.read_text()
    assert '"status": "ok"' in out_allowed.read_text()
