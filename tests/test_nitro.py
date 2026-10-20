"""Nitro groups must come out as [N+](=O)[O-] regardless of charge mode or atom order.

Left to itself, RDKit's xyz2mol returns a nitro group as N([O-])[O-], with the
-2 offset by charges invented elsewhere, or fails outright, depending on atom
order. The metronidazole and C2N2O5 geometries are the first conformers of the
records reported in pharmaforge issue #882.
"""

import pytest
from rdkit import Chem

import xyzrecover.core as core
from xyzrecover import PerceptionConfig, recover_xyz_text

# COMP6_Database/comp6_drugbank_testset/mol0657
METRONIDAZOLE_XYZ = """21
metronidazole (COMP6 drugbank mol0657)
O     -3.220097    -1.748696    -0.435234
C     -1.856780    -1.265697    -0.548208
C     -1.353898    -0.462068     0.670324
N     -0.209487     0.305716     0.258961
C      1.132778     0.161324    -0.190959
N      1.888617    -1.102983     0.068440
O      3.112546    -1.112296    -0.078843
O      1.294067    -2.118088     0.668943
C      1.599924     1.363430    -0.584952
N      0.742297     2.389111    -0.486717
C     -0.311990     1.738702     0.057782
C     -1.527225     2.677251     0.509458
H     -3.420547    -2.385402    -1.037830
H     -1.851239    -0.706685    -1.511960
H     -1.194922    -2.264781    -0.641510
H     -1.132532    -1.107505     1.620564
H     -2.116219     0.062991     1.163972
H      2.613645     1.455971    -1.092417
H     -1.375993     3.588754    -0.132718
H     -2.653333     2.013470     0.347194
H     -1.407833     2.755279     1.680327
"""

# QDPi2_Database/ani_qdpi/C2Cl0F0H0N2O5S0
C2N2O5_XYZ = """9
C2N2O5 (QDPi2 ani_qdpi)
C     -0.370563     0.085656     0.004305
C      1.759095    -0.226266    -0.080337
N     -1.761047    -0.160413     0.019973
N      0.092492     1.303312    -0.000303
O     -1.954815    -1.354748    -0.147303
O     -2.477999     0.809016     0.186795
O      1.450596     1.126830    -0.204521
O      2.841033    -0.649822     0.216954
O      0.559771    -0.825856    -0.012112
"""

TNT_XYZ = """21
TNT
C   -2.459528   0.001171  -0.294092
C   -0.962238   0.000985  -0.118332
C   -0.244819   1.180722   0.171807
N   -0.938251   2.449760   0.359998
O   -1.808940   2.490891   1.236738
O   -0.585762   3.389123  -0.362646
C    1.154113   1.203137   0.304239
C    1.867834   0.008208   0.167454
N    3.330418   0.013067   0.308227
O    3.878038   1.097901   0.541200
O    3.919164  -1.068653   0.188691
C    1.199621  -1.193997  -0.085840
C   -0.199956  -1.182508  -0.212756
N   -0.844841  -2.470491  -0.440664
O   -0.462220  -3.118665  -1.421651
O   -1.708119  -2.819552   0.372583
H   -2.952033  -0.164767   0.668797
H   -2.814897   0.944776  -0.720329
H   -2.782368  -0.771259  -0.999414
H    1.666821   2.142144   0.515753
H    1.747964  -2.131994  -0.179765
"""

NITROMETHANE_XYZ = """7
nitromethane
C    -0.6472   -0.0362   -0.0837
N     0.8266    0.0455    0.0993
O     1.4290   -1.0177    0.2900
O     1.3353    1.1722    0.0668
H    -0.8837   -0.9959   -0.5484
H    -0.9599    0.7867   -0.7301
H    -1.1001    0.0453    0.9061
"""

METHYL_NITRATE_XYZ = """8
methyl nitrate
C    -0.9838   -0.0104    0.3813
O     0.1756   -0.8158    0.2350
N     1.2945   -0.0745   -0.2138
O     1.1306    1.1302   -0.4140
O     2.3032   -0.7655   -0.3430
H    -0.8331    0.7690    1.1359
H    -1.2903    0.4220   -0.5772
H    -1.7968   -0.6550    0.7288
"""

NITRATE_XYZ = """4
nitrate
O     1.0734   -0.6230    0.1485
N     0.0000    0.0000    0.0960
O     0.0047    1.2395    0.0151
O    -1.0780   -0.6165    0.1245
"""


def _canon(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _permuted(xyz: str, order: list[int]) -> str:
    lines = xyz.splitlines()
    atom_lines = lines[2:]
    return "\n".join(lines[:2] + [atom_lines[i] for i in order]) + "\n"


@pytest.mark.parametrize("hueckel", [False, True])
def test_metronidazole_neutral(hueckel):
    rec = recover_xyz_text(
        METRONIDAZOLE_XYZ, PerceptionConfig(total_charge=0, use_hueckel=hueckel)
    )[0]
    assert rec.status == "ok"
    assert rec.charge == 0
    assert rec.smiles == _canon("Cc1ncc([N+](=O)[O-])n1CCO")
    assert rec.confidence == "high"


def test_metronidazole_unknown_charge_prefers_neutral_nitro():
    rec = recover_xyz_text(METRONIDAZOLE_XYZ)[0]
    assert rec.status == "ok"
    assert rec.charge == 0
    assert rec.smiles == _canon("Cc1ncc([N+](=O)[O-])n1CCO")


@pytest.mark.parametrize("config", [PerceptionConfig(total_charge=0), PerceptionConfig()])
def test_c2n2o5_has_no_invented_oxonium(config):
    rec = recover_xyz_text(C2N2O5_XYZ, config)[0]
    assert rec.status == "ok"
    assert rec.charge == 0
    assert rec.smiles == _canon("O=c1onc([N+](=O)[O-])o1")
    assert "[O+]" not in rec.smiles


def test_c2n2o5_declared_neutral_is_high_confidence():
    rec = recover_xyz_text(C2N2O5_XYZ, PerceptionConfig(total_charge=0))[0]
    assert rec.confidence == "high"


_TNT_N = 21
_TNT_ORDERS = {
    "identity": list(range(_TNT_N)),
    "reversed": list(range(_TNT_N))[::-1],
    "rotate 7": list(range(7, _TNT_N)) + list(range(7)),
    "rotate 13": list(range(13, _TNT_N)) + list(range(13)),
    "evens then odds": list(range(0, _TNT_N, 2)) + list(range(1, _TNT_N, 2)),
}


@pytest.mark.parametrize("order", _TNT_ORDERS.values(), ids=_TNT_ORDERS.keys())
def test_trinitro_result_does_not_depend_on_atom_order(order):
    # With fix_nitro_groups=False, RDKit fails on all but the reversed order here.
    rec = recover_xyz_text(_permuted(TNT_XYZ, order), PerceptionConfig(total_charge=0))[0]
    assert rec.status == "ok"
    assert rec.smiles == _canon("Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]")


@pytest.mark.parametrize(
    "xyz,charge,smiles",
    [
        (NITROMETHANE_XYZ, 0, "C[N+](=O)[O-]"),
        (METHYL_NITRATE_XYZ, 0, "CO[N+](=O)[O-]"),
        (NITRATE_XYZ, -1, "[O-][N+](=O)[O-]"),
    ],
)
def test_simple_nitro_species(xyz, charge, smiles):
    rec = recover_xyz_text(xyz, PerceptionConfig(total_charge=charge))[0]
    assert rec.status == "ok"
    assert rec.charge == charge
    assert rec.smiles == _canon(smiles)
    assert rec.confidence == "high"


@pytest.mark.parametrize(
    "config",
    [
        PerceptionConfig(total_charge=0, fix_nitro_groups=False),
        PerceptionConfig(total_charge=0, allow_charged_fragments=False),
    ],
    ids=["disabled", "radicals"],
)
def test_nitro_stand_in_not_used(monkeypatch, config):
    def fail(*args, **kwargs):
        raise AssertionError("nitro stand-in path should not run")

    monkeypatch.setattr(core, "_bond_orders_with_nitro_placeholders", fail)
    recs = recover_xyz_text(NITROMETHANE_XYZ, config)
    assert len(recs) == 1
    assert not any("nitro stand-in" in e for e in recs[0].errors)
