"""Group 1/2 metal counterions with ``split_fragments=False`` (issue #3).

The ionic fast-path used to act only on lone-atom fragments, so a salt perceived
whole reached the supported-element check with its metal still attached and came
back ``unsupported``. Whole mode now takes each group 1/2 metal out as its ion,
perceives the rest as one component, and merges the ions back into a single
record. The lone-atom tests cover the companion fix: a single atom used to ignore
its requested charge and pick up implicit hydrogens (a lone Cl came back as HCl).
"""

import pytest

from xyzrecover import PerceptionConfig, recover_xyz_block

WHOLE = PerceptionConfig(total_charge=0, split_fragments=False)

SODIUM_ACETATE = """\
8
sodium acetate
C   0.000  0.000  0.000
C   1.520  0.000  0.000
O   2.150  1.080  0.000
O   2.150 -1.080  0.000
H  -0.380  1.020  0.000
H  -0.380 -0.510  0.880
H  -0.380 -0.510 -0.880
Na  3.900  0.000  0.000
"""

# Sodium acetate with a water molecule 4 A away: two covalent fragments plus Na.
SODIUM_ACETATE_WATER = """\
11
sodium acetate + water
C   0.000  0.000  0.000
C   1.520  0.000  0.000
O   2.150  1.080  0.000
O   2.150 -1.080  0.000
H  -0.380  1.020  0.000
H  -0.380 -0.510  0.880
H  -0.380 -0.510 -0.880
Na  3.900  0.000  0.000
O  -4.000  0.000  0.000
H  -4.960  0.000  0.000
H  -3.760  0.930  0.000
"""


@pytest.mark.parametrize(
    "xyz, smiles",
    [
        ("2\nNaI\nNa 0 0 0\nI 0 0 2.71\n", "[I-].[Na+]"),
        ("2\nKBr\nK 0 0 0\nBr 0 0 2.82\n", "[Br-].[K+]"),
        ("2\nLiCl\nLi 0 0 0\nCl 0 0 2.02\n", "[Cl-].[Li+]"),
        ("2\nNaF\nNa 0 0 0\nF 0 0 1.93\n", "[F-].[Na+]"),
        (SODIUM_ACETATE, "CC(=O)[O-].[Na+]"),
    ],
)
def test_salt_perceived_whole(xyz, smiles):
    records = recover_xyz_block(xyz, WHOLE)
    assert len(records) == 1, "whole mode must keep one record per block"
    rec = records[0]
    assert rec.status == "ok"
    assert rec.smiles == smiles
    assert rec.charge == 0
    assert rec.confidence == "high"
    assert rec.inchikey  # one key for the whole salt
    assert sorted(rec.atom_indices) == list(range(len(rec.elements)))


def test_atom_indices_follow_the_record_mol():
    rec = recover_xyz_block("2\nNaI\nNa 0 0 0\nI 0 0 2.71\n", WHOLE)[0]
    # The anion's atoms come first, then the metals; each index names the
    # input atom at that position of the record's mol.
    assert rec.atom_indices == [1, 0]
    assert rec.elements == ["I", "Na"]
    assert [a.GetSymbol() for a in rec.mol.GetAtoms()] == ["I", "Na"]


def test_rest_of_block_stays_one_component():
    records = recover_xyz_block(SODIUM_ACETATE_WATER, WHOLE)
    assert len(records) == 1
    assert records[0].smiles == "CC(=O)[O-].O.[Na+]"
    assert records[0].charge == 0


def test_split_mode_unchanged():
    records = recover_xyz_block(SODIUM_ACETATE, PerceptionConfig(total_charge=0))
    assert sorted(r.smiles for r in records) == ["CC(=O)[O-]", "[Na+]"]


def test_whole_mode_without_ionic_recovery_stays_unsupported():
    cfg = PerceptionConfig(total_charge=0, split_fragments=False, recover_ionic_metals=False)
    records = recover_xyz_block(SODIUM_ACETATE, cfg)
    assert len(records) == 1
    assert records[0].status == "unsupported"


def test_hydrogen_bonded_only_to_the_metal_is_rejected():
    # The O-H hydrogen is stretched 2.4 A from its oxygen but sits 1.6 A from Na,
    # so the whole-block check counts it as bonded. Once Na is taken out it is
    # free, and must be rejected rather than perceived as [H+].
    xyz = """\
3
stretched NaOH
Na 0.000 0.000 0.000
O  2.000 0.000 0.000
H  0.400 1.550 0.000
"""
    records = recover_xyz_block(xyz, WHOLE)
    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].smiles is None
    assert "group 1/2 metal" in records[0].errors[0]


def test_sodium_hydride_is_an_ion_pair():
    rec = recover_xyz_block("2\nNaH\nNa 0 0 0\nH 0 0 1.89\n", WHOLE)[0]
    assert rec.smiles == "[H-].[Na+]"
    assert rec.charge == 0


@pytest.mark.parametrize(
    "symbol, charge, smiles", [("Cl", -1, "[Cl-]"), ("I", -1, "[I-]"), ("Cl", 0, "[Cl]")]
)
def test_lone_atom_takes_requested_charge_without_hydrogens(symbol, charge, smiles):
    rec = recover_xyz_block(f"1\nlone\n{symbol} 0 0 0\n", PerceptionConfig(total_charge=charge))[0]
    assert rec.smiles == smiles
    assert rec.charge == charge
    assert rec.mol.GetAtomWithIdx(0).GetTotalNumHs() == 0
