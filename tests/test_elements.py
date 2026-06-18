import pytest

from xyzrecover import DEFAULT_SUPPORTED_ELEMENTS, PerceptionConfig, recover_xyz_text


def _lone(sym: str) -> str:
    return f"1\n{sym}\n{sym} 0 0 0\n"


# --- ionic fast-path (default ON) -------------------------------------------


@pytest.mark.parametrize(
    "sym,smiles,charge",
    [
        ("Li", "[Li+]", 1),
        ("Na", "[Na+]", 1),
        ("K", "[K+]", 1),
        ("Mg", "[Mg+2]", 2),
        ("Ca", "[Ca+2]", 2),
    ],
)
def test_lone_group12_metal_recovered_as_ion(sym, smiles, charge):
    rec = recover_xyz_text(_lone(sym))[0]
    assert rec.status == "ok"
    assert rec.smiles == smiles
    assert rec.charge == charge
    assert rec.confidence == "high"
    assert rec.inchikey


def test_fastpath_beats_silent_hydride_even_with_guard_off():
    # Without the fast-path, RDKit returns [NaH]; the fast-path runs first.
    rec = recover_xyz_text(_lone("Na"), PerceptionConfig(restrict_to_supported_elements=False))[0]
    assert rec.smiles == "[Na+]"


def test_salt_charge_balances_via_total_charge():
    # Na+ ... OH-, declared neutral overall: DP must pick [Na+] and [OH-].
    xyz = "3\nNaOH ion pair\nNa 0 0 0\nO 4 0 0\nH 4 0 0.97\n"
    recs = recover_xyz_text(xyz, PerceptionConfig(total_charge=0))
    by_smiles = {r.smiles: r for r in recs}
    assert set(by_smiles) == {"[Na+]", "[OH-]"}
    assert by_smiles["[Na+]"].charge == 1
    assert by_smiles["[OH-]"].charge == -1


def test_water_and_ionic_metal_both_recover():
    xyz = """4
water and a distant sodium
O 0 0 0
H 0 0 0.96
H 0.92 0 -0.30
Na 10 10 10
"""
    recs = recover_xyz_text(xyz)
    assert len(recs) == 2
    assert sorted(r.smiles for r in recs) == ["O", "[Na+]"]
    assert all(r.status == "ok" for r in recs)


def test_per_fragment_charge_overrides_oxidation_state():
    # User explicitly declaring +2 for a lone Na is honored over the default +1.
    rec = recover_xyz_text(_lone("Na"), PerceptionConfig(per_fragment_charges=(2,)))[0]
    assert rec.status == "ok"
    assert rec.charge == 2


# --- guard still applies when the fast-path does not -------------------------


def test_group12_metal_unsupported_when_ionic_disabled():
    rec = recover_xyz_text(_lone("Na"), PerceptionConfig(recover_ionic_metals=False))[0]
    assert rec.status == "unsupported"
    assert rec.smiles is None


@pytest.mark.parametrize("sym", ["Fe", "Zn", "Cu"])
def test_transition_metal_lone_atom_is_unsupported(sym):
    # No unambiguous oxidation state -> not fast-pathed -> guarded.
    rec = recover_xyz_text(_lone(sym))[0]
    assert rec.status == "unsupported"
    assert rec.smiles is None


def test_multiatom_metal_fragment_is_unsupported():
    # NaCl as a single bonded fragment is not a lone ion.
    rec = recover_xyz_text(
        "2\nnacl\nNa 0 0 0\nCl 0 0 2.36\n", PerceptionConfig(split_fragments=False)
    )[0]
    assert rec.status == "unsupported"
    assert "Na" in rec.elements


def test_old_perception_path_when_both_disabled():
    rec = recover_xyz_text(
        _lone("Na"),
        PerceptionConfig(recover_ionic_metals=False, restrict_to_supported_elements=False),
    )[0]
    assert rec.status == "ok"
    assert rec.smiles is not None  # the (chemically wrong) [NaH]


def test_metal_excluded_from_default_supported_set():
    for sym in ["Li", "Na", "K", "Mg", "Ca", "Fe", "Zn"]:
        assert sym not in DEFAULT_SUPPORTED_ELEMENTS
    for sym in ["C", "H", "N", "O", "S", "Cl"]:
        assert sym in DEFAULT_SUPPORTED_ELEMENTS
