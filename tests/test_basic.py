from xyzrecover import PerceptionConfig, recover_xyz_text


def test_water():
    xyz = """3
water
O 0 0 0
H 0 0 0.96
H 0.92 0 -0.30
"""
    recs = recover_xyz_text(xyz)
    assert len(recs) == 1
    assert recs[0].status == "ok"
    assert recs[0].charge == 0
    assert recs[0].smiles == "O"
    assert recs[0].inchi.startswith("InChI=1S/H2O")


def test_ammonium_known_charge():
    xyz = """5
ammonium
N 0 0 0
H 0 0 1.04
H 0.98 0 -0.35
H -0.49 0.85 -0.35
H -0.49 -0.85 -0.35
"""
    recs = recover_xyz_text(xyz, PerceptionConfig(total_charge=1))
    assert len(recs) == 1
    assert recs[0].status == "ok"
    assert recs[0].charge == 1
    assert recs[0].smiles == "[NH4+]"


def test_two_fragments():
    xyz = """8
water methane
O 0 0 0
H 0 0 0.96
H 0.92 0 -0.30
C 5 0 0
H 6.09 0 0
H 4.64 1.03 0
H 4.64 -0.51 0.89
H 4.64 -0.51 -0.89
"""
    recs = recover_xyz_text(xyz)
    smiles = sorted(rec.smiles for rec in recs)
    assert smiles == ["C", "O"]
