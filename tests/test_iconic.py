"""Regression test: alkali-metal ions should perceive as bare cations.

xyz2mol (and thus xyzrecover) applies a covalent valence model to alkali /
alkaline-earth metals, so RDKit alone cannot represent an ionic metal correctly.
This case is sodium methyl phosphate — chemically a Na(+) counter-ion plus a
methyl phosphate(-1) anion — taken from an off-equilibrium snapshot of the QDPi2
"charged" dataset (DeepMD group C1H5N0O4...P1...Na1, first conformer).

The ionic fast-path (`recover_ionic_metals`, on by default) now separates the
Na from its close contact with the phosphate and recovers it as ``[Na+]``:
    [('COP(=O)(O)O', charge=0, confidence='high'),   # see below
     ('[Na+]',       charge=1, confidence='high')]

What remains unsolved (and is why this stays xfail): this is an off-equilibrium
*proton-transfer* snapshot in which both acidic protons are still bonded, so the
organic is geometrically the neutral diacid (C H5 O4 P, charge 0), not the -1
anion. A closed-shell -1 fragment is impossible without removing a proton that
is physically present, so the total comes to +1 rather than 0. Recovering the
intended protonation state from a mid-transfer geometry is a separate problem.
"""

import pytest

from xyzrecover import PerceptionConfig, recover_xyz_block

SODIUM_METHYL_PHOSPHATE_XYZ = """\
12
sodium methyl phosphate (off-equilibrium, QDPi2 charged)
O 2.318820 0.365010 1.045340
O 4.706880 1.433010 1.140340
O 3.254550 1.798520 -0.844460
O 4.323990 -0.459350 -0.381520
C 3.547300 -1.440280 -1.084270
P 3.532770 0.738590 0.313670
H 4.689490 1.133180 2.088970
H 4.088210 2.072550 -1.313450
H 4.233470 -2.241740 -1.405910
H 2.770090 -1.881050 -0.437530
H 3.061170 -1.013730 -1.977770
Na 0.767640 0.476850 1.546240
"""


@pytest.mark.xfail(
    reason="Na now recovers as [Na+] via the ionic fast-path, but this "
    "off-equilibrium proton-transfer snapshot still has both acidic protons "
    "bonded, so the organic is the neutral diacid (charge 0) not the -1 anion; "
    "the charges therefore sum to +1, not 0. Recovering the intended protonation "
    "state from a mid-transfer geometry is unsolved.",
    strict=True,
)
def test_alkali_metal_perceived_as_ion_pair():
    cfg = PerceptionConfig(
        total_charge=0,  # neutral salt: [Na+] + methyl phosphate(-1)
        split_fragments=True,
        allow_charged_fragments=True,
        use_hueckel=True,
    )
    records = recover_xyz_block(SODIUM_METHYL_PHOSPHATE_XYZ, cfg)
    assert records, "no records returned"

    smiles = [r.smiles for r in records]

    # 1) every fragment must perceive (today the Na fragment is smiles=None)
    assert all(smiles), f"a fragment failed to perceive: {smiles}"

    # 2) sodium must be a bare cation, never a fabricated hydride
    assert "[Na+]" in smiles, f"sodium not perceived as [Na+]: {smiles}"
    assert not any("Na" in s and s != "[Na+]" for s in smiles), (
        f"sodium bonded into a molecular fragment (e.g. [NaH]): {smiles}"
    )

    # 3) charge must localize correctly: [Na+] (+1) + phosphate anion (-1) = 0
    charges = [r.charge for r in records]
    assert sum(c or 0 for c in charges) == 0
    assert any(c == 1 for c in charges), f"no +1 cation fragment: {charges}"
    assert any(c == -1 for c in charges), f"organic not anionic (got neutral acid): {charges}"


def test_alkali_metal_recovered_as_cation():
    """The metal-recovery half of the case above, which the fast-path fixes.

    Independent of the (unsolved) protonation-state question: the Na counter-ion
    must always come back as a bare [Na+], never None and never a fabricated
    hydride, and every fragment must perceive.
    """
    cfg = PerceptionConfig(total_charge=0, use_hueckel=True)
    records = recover_xyz_block(SODIUM_METHYL_PHOSPHATE_XYZ, cfg)
    smiles = [r.smiles for r in records]

    assert all(smiles), f"a fragment failed to perceive: {smiles}"
    assert "[Na+]" in smiles, f"sodium not perceived as [Na+]: {smiles}"
    assert not any(s and "Na" in s and s != "[Na+]" for s in smiles), (
        f"sodium bonded into a molecular fragment (e.g. [NaH]): {smiles}"
    )
    na = next(r for r in records if r.smiles == "[Na+]")
    assert na.charge == 1
    assert na.status == "ok"
