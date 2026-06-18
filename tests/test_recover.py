import json

import pytest
from rdkit import Chem

from xyzrecover import (
    PerceptionConfig,
    XyzRecoverError,
    records_to_json,
    recover_xyz_text,
    write_csv,
    write_sdf,
)

WATER = """3
water
O 0 0 0
H 0 0 0.96
H 0.92 0 -0.30
"""

WATER_METHANE = """8
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

HYDROXIDE = """2
hydroxide
O 0 0 0
H 0 0 0.97
"""

ALLOWED_CONFIDENCE = {"high", "medium", "low", "none"}


def test_water_record_fields():
    rec = recover_xyz_text(WATER)[0]
    assert rec.status == "ok"
    assert rec.charge == 0
    assert rec.formula == "H2O"
    assert rec.smiles == "O"
    assert "H" in rec.explicit_h_smiles
    assert rec.inchikey and rec.inchikey.startswith("XLYOFNOQVPJJNP")  # water InChIKey
    assert rec.confidence in ALLOWED_CONFIDENCE


def test_keep_explicit_h_smiles_uses_explicit_as_primary():
    rec = recover_xyz_text(WATER, PerceptionConfig(keep_explicit_h_smiles=True))[0]
    assert rec.smiles == rec.explicit_h_smiles
    assert "H" in rec.smiles


def test_fragment_split_separates_components():
    recs = recover_xyz_text(WATER_METHANE)
    assert len(recs) == 2
    assert [r.molecule_index for r in recs] == [0, 1]
    assert sorted(r.smiles for r in recs) == ["C", "O"]
    # atom_indices partition all 8 atoms exactly once.
    covered = sorted(i for r in recs for i in r.atom_indices)
    assert covered == list(range(8))


def test_no_split_keeps_one_record_with_dot_smiles():
    recs = recover_xyz_text(WATER_METHANE, PerceptionConfig(split_fragments=False))
    assert len(recs) == 1
    assert "." in recs[0].smiles  # disconnected components in a single SMILES


def test_known_total_charge_hydroxide():
    rec = recover_xyz_text(HYDROXIDE, PerceptionConfig(total_charge=-1))[0]
    assert rec.status == "ok"
    assert rec.charge == -1
    assert rec.smiles == "[OH-]"
    assert rec.confidence == "high"


def test_per_fragment_charges_length_mismatch_raises():
    with pytest.raises(XyzRecoverError):
        recover_xyz_text(WATER_METHANE, PerceptionConfig(per_fragment_charges=(0,)))


def test_per_fragment_charges_applied():
    recs = recover_xyz_text(WATER_METHANE, PerceptionConfig(per_fragment_charges=(0, 0)))
    assert all(r.status == "ok" for r in recs)
    assert all(r.charge == 0 for r in recs)


def test_impossible_asserted_charge_fails_without_fallback():
    rec = recover_xyz_text(WATER, PerceptionConfig(per_fragment_charges=(7,)))[0]
    assert rec.status == "failed"
    assert rec.charge is None
    assert rec.errors  # at least one failed attempt recorded


def test_charge_fallback_recovers_with_warning():
    rec = recover_xyz_text(
        WATER, PerceptionConfig(per_fragment_charges=(7,), charge_fallback=True)
    )[0]
    assert rec.status == "ok"
    assert rec.charge == 0
    assert rec.smiles == "O"
    assert any("fallback" in w.lower() for w in rec.warnings)


def test_to_dict_excludes_mol_by_default():
    rec = recover_xyz_text(WATER)[0]
    assert rec.mol is not None
    data = rec.to_dict()
    assert "mol" not in data
    assert rec.to_dict(include_mol=True)["mol"] is rec.mol


def test_records_to_json_is_valid_and_sorted():
    recs = recover_xyz_text(WATER_METHANE)
    payload = json.loads(records_to_json(recs))
    assert isinstance(payload, list) and len(payload) == 2
    for entry in payload:
        assert {"smiles", "charge", "inchikey", "confidence", "status"} <= entry.keys()
        assert "mol" not in entry


def test_write_csv_and_sdf(tmp_path):
    recs = recover_xyz_text(WATER_METHANE)
    csv_path = tmp_path / "out.csv"
    sdf_path = tmp_path / "out.sdf"
    write_csv(recs, csv_path)
    write_sdf(recs, sdf_path)

    csv_text = csv_path.read_text()
    assert "smiles" in csv_text.splitlines()[0]
    assert len(csv_text.splitlines()) == 1 + len(recs)  # header + rows

    supplier = Chem.SDMolSupplier(str(sdf_path))
    mols = [m for m in supplier if m is not None]
    assert len(mols) == 2
    assert all(m.HasProp("smiles") for m in mols)
