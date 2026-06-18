import pytest

from xyzrecover import XyzRecoverError, parse_xyz_blocks, recover_xyz_text

STANDARD = """3
water
O 0 0 0
H 0 0 0.96
H 0.92 0 -0.30
"""

HEADERLESS = """O 0 0 0
H 0 0 0.96
H 0.92 0 -0.30
"""

EXTRA_COLUMNS = """3
water with forces
O 0 0 0     0.1 0.2 0.3
H 0 0 0.96  0.0 0.0 0.0
H 0.92 0 -0.30  0.0 0.0 0.0
"""


def test_standard_single_block():
    blocks = parse_xyz_blocks(STANDARD)
    assert len(blocks) == 1


def test_headerless_block_parses():
    blocks = parse_xyz_blocks(HEADERLESS)
    assert len(blocks) == 1
    recs = recover_xyz_text(HEADERLESS)
    assert len(recs) == 1
    assert recs[0].smiles == "O"


def test_extra_columns_are_ignored():
    recs = recover_xyz_text(EXTRA_COLUMNS)
    assert len(recs) == 1
    assert recs[0].smiles == "O"


def test_multi_frame_yields_one_set_per_frame():
    text = STANDARD + STANDARD
    blocks = parse_xyz_blocks(text)
    assert len(blocks) == 2
    recs = recover_xyz_text(text)
    assert [r.block_index for r in recs] == [0, 1]
    assert all(r.smiles == "O" for r in recs)


def test_leading_and_trailing_blank_lines():
    text = "\n\n" + STANDARD + "\n\n"
    blocks = parse_xyz_blocks(text)
    assert len(blocks) == 1


def test_atom_count_mismatch_raises():
    bad = "3\nwater\nO 0 0 0\nH 0 0 0.96\n"  # declares 3 atoms, supplies 2
    with pytest.raises(XyzRecoverError):
        parse_xyz_blocks(bad)


def test_unparseable_content_raises():
    with pytest.raises(XyzRecoverError):
        parse_xyz_blocks("this is not xyz at all\n")
