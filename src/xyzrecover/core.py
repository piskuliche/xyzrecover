from __future__ import annotations

import copy
import csv
import itertools
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdDetermineBonds, rdMolDescriptors
from rdkit.Chem import inchi as rd_inchi
from rdkit.Geometry import Point3D


class XyzRecoverError(RuntimeError):
    """Raised when XYZ parsing or molecule recovery cannot proceed."""


# Main-group elements that RDKit's xyz2mol / DetermineBonds valence model can
# handle. Anything outside this set (metals, noble gases, most of the periodic
# table) is not reliably perceivable: RDKit may raise, or worse, silently return
# a wrong structure at high confidence (e.g. a lone Na+ comes back as ``[NaH]``).
# Such fragments are reported with status "unsupported" rather than perceived.
DEFAULT_SUPPORTED_ELEMENTS: frozenset[str] = frozenset(
    {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}
)

# Group 1 / group 2 metals have an unambiguous ionic oxidation state when they
# appear as an isolated single-atom fragment (a counterion in a salt or ion
# pair): alkali metals are +1, alkaline-earth metals are +2. These can be
# recovered directly as their ion instead of being sent through RDKit (which
# would mis-assign them, see DEFAULT_SUPPORTED_ELEMENTS).
ALKALI_METALS: frozenset[str] = frozenset({"Li", "Na", "K", "Rb", "Cs", "Fr"})
ALKALINE_EARTH_METALS: frozenset[str] = frozenset({"Be", "Mg", "Ca", "Sr", "Ba", "Ra"})


@dataclass(frozen=True)
class PerceptionConfig:
    """Configuration for recovering molecules from XYZ coordinates.

    Parameters
    ----------
    total_charge
        Optional total charge for the whole XYZ block. If multiple disconnected
        covalent components are present, the recovered fragment charges are
        constrained to sum to this value.
    per_fragment_charges
        Optional explicit charge per disconnected component. This overrides
        `candidate_charges` for the corresponding components. The order matches
        the order in which connectivity perception discovers the components,
        which is the same order as ``molecule_index`` in the returned records
        (not necessarily the atom order in the input). Inspect a first
        unconstrained run to see how a structure splits before relying on this.
    candidate_charges
        Charge states to try when a component charge is unknown.
    charge_fallback
        If True and a charge-constrained component (via `total_charge` for a
        single component, or `per_fragment_charges`) yields no valid assignment,
        retry that component using `candidate_charges`. The recovered record is
        flagged with a warning. Default False, so an asserted charge that cannot
        be satisfied fails loudly rather than silently guessing.
    restrict_to_supported_elements
        If True (default), a component containing any element RDKit's xyz2mol
        valence model cannot handle (metals, noble gases) is reported with
        ``status="unsupported"`` and ``smiles=None`` instead of being perceived.
        This prevents silent, high-confidence garbage such as a lone Na+ being
        recovered as ``[NaH]``. Set False to attempt perception anyway.
    supported_elements
        The set of element symbols treated as supported. ``None`` uses
        `DEFAULT_SUPPORTED_ELEMENTS`. Extend it if your RDKit build handles more.
    recover_ionic_metals
        If True (default), an isolated single-atom fragment of a group 1 or
        group 2 metal (a salt/ion-pair counterion) is recovered directly as its
        ion (``[Na+]``, ``[Ca+2]``, …) using the group oxidation state, bypassing
        RDKit. This runs before the unsupported-element check. A declared charge
        (`per_fragment_charges`, or `total_charge` for a lone atom) overrides the
        default oxidation state. Multi-atom metal fragments and other metals are
        not affected.
    fix_nitro_groups
        If True (default), each nitro-type group (an N with three bonds, two of
        them to terminal O atoms: nitro, nitrate, nitramine) is assigned the
        charge-separated ``[N+](=O)[O-]`` form, and RDKit assigns bond orders
        only to the rest of the component, with the group replaced by a neutral
        monovalent stand-in. Without this, RDKit's xyz2mol tends to return
        ``N([O-])[O-]`` balanced by invented charges elsewhere (``[O+]``,
        ``[S+]``), or to fail outright, depending on atom order. Only applies
        when `allow_charged_fragments` is True. Nitronate anions come out as
        the carbanion resonance form (``[CH2-][N+](=O)[O-]``), which has the
        same InChI.
    reject_isolated_hydrogens
        If True (default), a block of more than one atom in which connectivity
        perception leaves any hydrogen with no bonded neighbour is reported as
        failed (``status="failed"``, ``smiles=None``) for every component, with
        an error naming the hydrogen's atom index. Such a hydrogen is almost
        always an X-H bond stretched past the connectivity cutoff (normal-mode
        or MD sampling). Left alone, bond-order assignment balances it as a
        free ``[H+]`` against a carbanion or alkoxide, which a declared total
        charge then accepts at high confidence. The record is failed rather
        than repaired, because reattaching the hydrogen to its nearest heavy
        atom would silently mis-assign real proton-transfer geometries; try
        another conformer instead. ``H2`` and a single-atom block are not
        affected.
    split_fragments
        If True, split disconnected covalent components before assigning bond
        orders. This is recommended for XYZ files containing multiple molecules.
    allow_charged_fragments
        RDKit xyz2mol option. True places formal charges to satisfy valence;
        False prefers radical electrons instead.
    use_hueckel
        Use extended Hueckel for connectivity when available in the RDKit build.
    use_vdw
        Use the van der Waals connectivity method rather than connect-the-dots.
    cov_factor
        Covalent-radius multiplier used for connectivity perception.
    embed_chiral
        Ask RDKit to embed 3D-derived chirality while assigning bonds.
    sanitize
        Run ``Chem.SanitizeMol`` on each perceived candidate. A candidate that
        fails sanitization is rejected. Applied independently of `embed_chiral`.
    keep_explicit_h_smiles
        If True, the primary SMILES retains explicit hydrogens from the XYZ.
        Otherwise, a conventional implicit-H canonical SMILES is generated when
        possible; explicit_h_smiles is always retained.
    max_iterations
        RDKit maxIterations for DetermineBonds. Zero means RDKit default/no cap.
    max_reported_alternates
        Number of non-selected candidate assignments to include in records.
    """

    total_charge: int | None = None
    per_fragment_charges: tuple[int, ...] | None = None
    candidate_charges: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
    charge_fallback: bool = False
    restrict_to_supported_elements: bool = True
    supported_elements: frozenset[str] | None = None
    recover_ionic_metals: bool = True
    fix_nitro_groups: bool = True
    reject_isolated_hydrogens: bool = True
    split_fragments: bool = True
    allow_charged_fragments: bool = True
    use_hueckel: bool = False
    use_vdw: bool = False
    cov_factor: float = 1.30
    embed_chiral: bool = True
    sanitize: bool = True
    keep_explicit_h_smiles: bool = False
    max_iterations: int = 0
    max_reported_alternates: int = 5


@dataclass
class MoleculeRecord:
    """Serializable result for one recovered component."""

    source: str | None
    block_index: int
    molecule_index: int
    atom_indices: list[int]
    elements: list[str]
    formula: str | None
    charge: int | None
    smiles: str | None
    explicit_h_smiles: str | None
    inchi: str | None
    inchikey: str | None
    confidence: str
    status: str
    score: dict[str, int | float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    alternates: list[dict[str, Any]] = field(default_factory=list)
    mol: Any | None = field(default=None, repr=False, compare=False)

    def to_dict(self, include_mol: bool = False) -> dict[str, Any]:
        # Build the dict directly from fields, deep-copying only the plain data.
        # Using dataclasses.asdict here would deep-copy the RDKit Mol (an
        # expensive, sometimes-failing operation) on every call, only to discard
        # it. to_dict() runs once per record for JSON, CSV, and SDF output.
        data = {
            f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self) if f.name != "mol"
        }
        if include_mol:
            data["mol"] = self.mol
        return data


@dataclass
class _AtomRow:
    symbol: str
    x: float
    y: float
    z: float


@dataclass
class _Candidate:
    charge: int
    mol: Chem.Mol
    smiles: str
    explicit_h_smiles: str
    inchi: str | None
    inchikey: str | None
    formula: str | None
    score_tuple: tuple[int | float, ...]
    score: dict[str, int | float]
    warnings: list[str]


def _periodic_table() -> Chem.PeriodicTable:
    return Chem.GetPeriodicTable()


def _normalise_symbol(token: str) -> str | None:
    token = token.strip()
    if not token:
        return None
    pt = _periodic_table()
    if token.isdigit():
        anum = int(token)
        if anum <= 0 or anum > 118:
            return None
        return pt.GetElementSymbol(anum)
    symbol = token[0].upper() + token[1:].lower()
    try:
        if pt.GetAtomicNumber(symbol) <= 0:
            return None
    except Exception:
        return None
    return symbol


def _parse_atom_line(line: str) -> _AtomRow | None:
    parts = line.split()
    if len(parts) < 4:
        return None
    symbol = _normalise_symbol(parts[0])
    if symbol is None:
        return None
    try:
        x, y, z = map(float, parts[1:4])
    except ValueError:
        return None
    return _AtomRow(symbol=symbol, x=x, y=y, z=z)


def _format_xyz(atoms: Sequence[_AtomRow], comment: str = "") -> str:
    lines = [str(len(atoms)), comment.rstrip("\n")]
    for atom in atoms:
        lines.append(f"{atom.symbol:<3s} {atom.x: .10f} {atom.y: .10f} {atom.z: .10f}")
    return "\n".join(lines) + "\n"


def _atoms_from_xyz_block(block: str) -> tuple[list[_AtomRow], str]:
    lines = block.splitlines()
    if not lines:
        raise XyzRecoverError("Empty XYZ block")
    first = lines[0].split()
    comment = ""
    atom_lines: list[str]
    try:
        n_atoms = int(first[0])
        comment = lines[1] if len(lines) > 1 else ""
        atom_lines = lines[2 : 2 + n_atoms]
        if len(atom_lines) != n_atoms:
            raise XyzRecoverError(
                f"XYZ block declares {n_atoms} atoms but contains {len(atom_lines)} atom lines"
            )
    except (ValueError, IndexError):
        atom_lines = lines

    atoms: list[_AtomRow] = []
    for line in atom_lines:
        if not line.strip():
            continue
        parsed = _parse_atom_line(line)
        if parsed is None:
            raise XyzRecoverError(f"Cannot parse XYZ atom line: {line!r}")
        atoms.append(parsed)
    if not atoms:
        raise XyzRecoverError("XYZ block contains no atoms")
    return atoms, comment


def parse_xyz_blocks(text: str) -> list[str]:
    """Parse one or more standard XYZ blocks from text.

    The parser accepts normal XYZ blocks (`natoms`, comment, atom lines),
    multi-frame XYZ files, and a headerless single block containing only atom
    rows. Extra columns after x/y/z are ignored.
    """
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break

        parts = lines[i].split()
        n_atoms: int | None = None
        try:
            n_atoms = int(parts[0])
        except (ValueError, IndexError):
            n_atoms = None

        if n_atoms is not None:
            # Standard XYZ: atom count, comment, N atom lines. If the comment
            # line is missing, fall back to treating the next N lines as atoms.
            with_comment = lines[i + 2 : i + 2 + n_atoms]
            if len(with_comment) == n_atoms and all(_parse_atom_line(x) for x in with_comment):
                atoms = [_parse_atom_line(x) for x in with_comment]
                assert all(a is not None for a in atoms)
                blocks.append(
                    _format_xyz(
                        [a for a in atoms if a is not None],
                        lines[i + 1] if i + 1 < len(lines) else "",
                    )
                )
                i += n_atoms + 2
                continue
            no_comment = lines[i + 1 : i + 1 + n_atoms]
            if len(no_comment) == n_atoms and all(_parse_atom_line(x) for x in no_comment):
                atoms = [_parse_atom_line(x) for x in no_comment]
                assert all(a is not None for a in atoms)
                blocks.append(_format_xyz([a for a in atoms if a is not None], ""))
                i += n_atoms + 1
                continue
            raise XyzRecoverError(f"Could not parse XYZ block starting at line {i + 1}")

        # Headerless block: consume consecutive atom lines.
        atoms: list[_AtomRow] = []
        start = i
        while i < len(lines):
            if not lines[i].strip():
                if atoms:
                    break
                i += 1
                continue
            atom = _parse_atom_line(lines[i])
            if atom is None:
                if atoms:
                    break
                raise XyzRecoverError(f"Expected XYZ atom row at line {i + 1}: {lines[i]!r}")
            atoms.append(atom)
            i += 1
        if not atoms:
            raise XyzRecoverError(f"Could not parse XYZ content near line {start + 1}")
        blocks.append(_format_xyz(atoms, "headerless XYZ"))
    return blocks


def _mol_from_xyz_block(block: str) -> Chem.Mol:
    mol = Chem.MolFromXYZBlock(block)
    if mol is None:
        raise XyzRecoverError("RDKit could not parse XYZ block")
    if mol.GetNumConformers() == 0:
        raise XyzRecoverError("XYZ molecule has no 3D conformer")
    return mol


def _manual_components(atoms: Sequence[_AtomRow], cov_factor: float) -> list[tuple[int, ...]]:
    pt = _periodic_table()
    n = len(atoms)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        ai = pt.GetAtomicNumber(atoms[i].symbol)
        ri = pt.GetRcovalent(ai)
        for j in range(i + 1, n):
            aj = pt.GetAtomicNumber(atoms[j].symbol)
            rj = pt.GetRcovalent(aj)
            dx = atoms[i].x - atoms[j].x
            dy = atoms[i].y - atoms[j].y
            dz = atoms[i].z - atoms[j].z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            # Ignore near-duplicate coordinates; otherwise use a covalent-radius
            # cutoff similar in spirit to RDKit's connectivity perception.
            if 0.1 < dist <= cov_factor * (ri + rj):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)
    return [tuple(v) for v in groups.values()]


def _ionic_metal_oxidation_state(symbol: str) -> int | None:
    """Default ionic oxidation state for a lone group 1/2 metal, else None."""
    if symbol in ALKALI_METALS:
        return 1
    if symbol in ALKALINE_EARTH_METALS:
        return 2
    return None


def _separate_ionic_metals(
    frags: Sequence[Sequence[int]], atoms: Sequence[_AtomRow], config: PerceptionConfig
) -> list[tuple[int, ...]]:
    """Pull group 1/2 metal atoms out of covalent fragments into singletons.

    Connectivity perception treats a short metal...ligand contact (a normal
    ionic distance, ~1.6-2.5 A) as a covalent bond, merging an ionic counterion
    into its anion. Alkali / alkaline-earth metals are ionic in these contexts,
    so extract each into its own fragment; the fast-path then recovers it as an
    ion and the remaining atoms perceive cleanly as the anion.
    """
    if not config.recover_ionic_metals:
        return [tuple(frag) for frag in frags]
    rest: list[tuple[int, ...]] = []
    metal_singletons: list[tuple[int, ...]] = []
    for frag in frags:
        kept = tuple(i for i in frag if _ionic_metal_oxidation_state(atoms[i].symbol) is None)
        if kept:
            rest.append(kept)
        metal_singletons.extend(
            (i,) for i in frag if _ionic_metal_oxidation_state(atoms[i].symbol) is not None
        )
    return rest + metal_singletons


def _split_components(
    block: str, atoms: Sequence[_AtomRow], config: PerceptionConfig
) -> tuple[list[tuple[int, ...]], list[str]]:
    warnings: list[str] = []
    if not config.split_fragments:
        return [tuple(range(len(atoms)))], warnings

    mol = _mol_from_xyz_block(block)
    use_hueckel = config.use_hueckel
    if use_hueckel and not rdDetermineBonds.hueckelEnabled():
        warnings.append("RDKit was not built with YAeHMOP; falling back from Hueckel connectivity.")
        use_hueckel = False

    try:
        rdDetermineBonds.DetermineConnectivity(
            mol,
            useHueckel=use_hueckel,
            charge=config.total_charge or 0,
            covFactor=config.cov_factor,
            useVdw=config.use_vdw,
        )
        frags = [tuple(int(i) for i in frag) for frag in Chem.GetMolFrags(mol, asMols=False)]
        if frags:
            return _separate_ionic_metals(frags, atoms, config), warnings
    except Exception as exc:  # pragma: no cover - fallback path depends on RDKit build/inputs
        warnings.append(f"RDKit connectivity failed; used covalent-radius fallback: {exc}")
    return _separate_ionic_metals(
        _manual_components(atoms, config.cov_factor), atoms, config
    ), warnings


def _isolated_hydrogens(
    block: str, atoms: Sequence[_AtomRow], config: PerceptionConfig
) -> list[int]:
    """Indices of hydrogen atoms that connectivity perception leaves unbonded.

    Uses the same connectivity call (and covalent-radius fallback) as fragment
    splitting, on the whole block, so the answer does not depend on
    `split_fragments`. A single-atom block has no isolated hydrogen by definition.
    """
    if len(atoms) < 2 or not any(atom.symbol == "H" for atom in atoms):
        return []
    use_hueckel = config.use_hueckel and rdDetermineBonds.hueckelEnabled()
    try:
        mol = _mol_from_xyz_block(block)
        rdDetermineBonds.DetermineConnectivity(
            mol,
            useHueckel=use_hueckel,
            charge=config.total_charge or 0,
            covFactor=config.cov_factor,
            useVdw=config.use_vdw,
        )
        bonded = [atom.GetDegree() > 0 for atom in mol.GetAtoms()]
    except Exception:  # pragma: no cover - fallback path depends on RDKit build/inputs
        bonded = [False] * len(atoms)
        for component in _manual_components(atoms, config.cov_factor):
            if len(component) > 1:
                for idx in component:
                    bonded[idx] = True
    return [i for i, atom in enumerate(atoms) if atom.symbol == "H" and not bonded[i]]


def _fragment_xyz(
    atoms: Sequence[_AtomRow], indices: Sequence[int], comment: str = ""
) -> tuple[str, list[str]]:
    selected = [atoms[i] for i in indices]
    frag_comment = comment or "fragment"
    return _format_xyz(selected, frag_comment), [a.symbol for a in selected]


def _unique_ordered(values: Iterable[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(int(value))
    return tuple(ordered)


def _charge_search_order(charges: Iterable[int]) -> tuple[int, ...]:
    # Prefer lower absolute charge in unconstrained searches, but keep both signs.
    return tuple(sorted(_unique_ordered(charges), key=lambda x: (abs(x), x)))


def _charges_for_fragment(
    config: PerceptionConfig, n_fragments: int, frag_index: int
) -> tuple[int, ...]:
    if config.per_fragment_charges is not None:
        if frag_index >= len(config.per_fragment_charges):
            raise XyzRecoverError(
                "per_fragment_charges has fewer entries than the number of disconnected components"
            )
        return (int(config.per_fragment_charges[frag_index]),)
    if config.total_charge is not None and n_fragments == 1:
        return (int(config.total_charge),)
    return _charge_search_order(config.candidate_charges)


def _safe_remove_hs(mol: Chem.Mol) -> Chem.Mol:
    try:
        return Chem.RemoveHs(mol, sanitize=True)
    except Exception:
        try:
            return Chem.RemoveHs(mol, sanitize=False)
        except Exception:
            return Chem.Mol(mol)


def _nitro_groups(mol: Chem.Mol) -> list[tuple[int, int, int]]:
    """Find nitro-type groups as ``(n, o_double, o_single)`` atom indices.

    A nitro-type group is an N with three bonds, at least two of them to
    terminal O atoms (nitro, nitrate, nitramine). The two lowest-index terminal
    O atoms are used, so the assignment is deterministic.
    """
    groups: list[tuple[int, int, int]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetDegree() != 3:
            continue
        terminal_o = sorted(
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetAtomicNum() == 8 and nbr.GetDegree() == 1
        )
        if len(terminal_o) >= 2:
            groups.append((atom.GetIdx(), terminal_o[0], terminal_o[1]))
    return groups


def _bond_orders_with_nitro_placeholders(
    mol: Chem.Mol,
    groups: Sequence[tuple[int, int, int]],
    charge: int,
    config: PerceptionConfig,
) -> Chem.Mol:
    """Assign bond orders to a connected `mol` with each nitro group as ``[N+](=O)[O-]``.

    RDKit's xyz2mol tries nitrogen at valence 3 before 4 and keeps the first
    charge-balanced assignment, so a nitro N tends to come out as N([O-])[O-]
    with the -2 offset by charges invented elsewhere. A charge-bookkeeping
    quirk for three-bonded carbons can also make it abort the search, which
    depends on atom order. A nitro group is net-neutral and singly bonded to
    its parent, so perceive the rest of the component with the group replaced
    by a neutral F, then write the nitro group back.
    """
    removed = {o for _, o_double, o_single in groups for o in (o_double, o_single)}
    kept = [i for i in range(mol.GetNumAtoms()) if i not in removed]

    work = Chem.RWMol(mol)
    # Hueckel connectivity leaves the total charge on atom 0; start clean.
    for atom in work.GetAtoms():
        atom.SetFormalCharge(0)
    for n, _, _ in groups:
        work.GetAtomWithIdx(n).SetAtomicNum(9)
    for idx in sorted(removed, reverse=True):
        work.RemoveAtom(idx)
    work_mol = work.GetMol()
    rdDetermineBonds.DetermineBondOrders(
        work_mol,
        charge=int(charge),
        allowChargedFragments=config.allow_charged_fragments,
        embedChiral=False,
        useAtomMap=False,
        maxIterations=int(config.max_iterations),
    )

    out = Chem.RWMol(mol)
    for atom in out.GetAtoms():
        atom.SetFormalCharge(0)
    for bond in work_mol.GetBonds():
        out.GetBondBetweenAtoms(
            kept[bond.GetBeginAtomIdx()], kept[bond.GetEndAtomIdx()]
        ).SetBondType(bond.GetBondType())
    for atom in work_mol.GetAtoms():
        target = out.GetAtomWithIdx(kept[atom.GetIdx()])
        target.SetFormalCharge(atom.GetFormalCharge())
        target.SetNumRadicalElectrons(atom.GetNumRadicalElectrons())
    for n, o_double, o_single in groups:
        out.GetAtomWithIdx(n).SetFormalCharge(1)
        out.GetAtomWithIdx(o_single).SetFormalCharge(-1)
        out.GetBondBetweenAtoms(n, o_double).SetBondType(Chem.BondType.DOUBLE)
    result = out.GetMol()
    if config.embed_chiral:
        # Same steps DetermineBondOrders runs for embedChiral=True.
        Chem.SanitizeMol(result)
        Chem.SetDoubleBondNeighborDirections(result, result.GetConformer())
        Chem.AssignStereochemistryFrom3D(result)
    return result


def _determine_bonds(
    mol: Chem.Mol, charge: int, use_hueckel: bool, config: PerceptionConfig
) -> Chem.Mol:
    """``rdDetermineBonds.DetermineBonds``, with nitro groups fixed before bond orders."""
    if mol.GetNumAtoms() <= 1:
        # DetermineBonds leaves a single atom untouched: it ignores the requested
        # charge, and sanitizing then fills the valence with implicit hydrogens
        # (a lone Cl came back as HCl, and NaI as "I.[Na+]"). A lone atom has no
        # hydrogens to add, so it is the bare atom at the requested charge.
        rwmol = Chem.RWMol(mol)
        for atom in rwmol.GetAtoms():
            atom.SetFormalCharge(int(charge))
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
        return rwmol.GetMol()
    rdDetermineBonds.DetermineConnectivity(
        mol,
        useHueckel=use_hueckel,
        charge=int(charge),
        covFactor=config.cov_factor,
        useVdw=config.use_vdw,
    )
    if config.fix_nitro_groups and config.allow_charged_fragments:
        groups = _nitro_groups(mol)
        if groups:
            return _bond_orders_with_nitro_placeholders(mol, groups, charge, config)
    rdDetermineBonds.DetermineBondOrders(
        mol,
        charge=int(charge),
        allowChargedFragments=config.allow_charged_fragments,
        embedChiral=config.embed_chiral,
        useAtomMap=False,
        maxIterations=int(config.max_iterations),
    )
    return mol


def _candidate_from_mol(mol: Chem.Mol, charge: int, config: PerceptionConfig) -> _Candidate:
    warnings: list[str] = []
    formula: str | None = None
    inchi_value: str | None = None
    inchikey_value: str | None = None

    formal_charge = Chem.GetFormalCharge(mol)
    radical_electrons = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    atom_charge_abs_sum = sum(abs(atom.GetFormalCharge()) for atom in mol.GetAtoms())
    num_atom_charges = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0)
    disconnected = max(0, len(Chem.GetMolFrags(mol, asMols=False)) - 1)

    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception as exc:
        warnings.append(f"Could not calculate formula: {exc}")

    explicit_h_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    smiles_mol = Chem.Mol(mol) if config.keep_explicit_h_smiles else _safe_remove_hs(mol)
    smiles = Chem.MolToSmiles(smiles_mol, isomericSmiles=True, canonical=True)

    try:
        with rdBase.BlockLogs():
            inchi_value = rd_inchi.MolToInchi(mol, logLevel=None)
    except Exception as exc:
        warnings.append(f"InChI generation failed: {exc}")
    try:
        with rdBase.BlockLogs():
            inchikey_value = rd_inchi.MolToInchiKey(mol)
    except Exception as exc:
        warnings.append(f"InChIKey generation failed: {exc}")

    molecular_weight = float(Descriptors.MolWt(mol))
    # Tuple order intentionally ranks chemically conservative candidates first.
    score_tuple: tuple[int | float, ...] = (
        1 if inchi_value is None else 0,
        radical_electrons,
        abs(formal_charge),
        atom_charge_abs_sum,
        num_atom_charges,
        disconnected,
        molecular_weight / 100000.0,  # deterministic tiny tie-breaker
    )
    score = {
        "inchi_failed": 1 if inchi_value is None else 0,
        "radical_electrons": radical_electrons,
        "abs_total_charge": abs(formal_charge),
        "atom_charge_abs_sum": atom_charge_abs_sum,
        "num_charged_atoms": num_atom_charges,
        "extra_disconnected_fragments": disconnected,
    }
    if formal_charge != charge:
        warnings.append(
            f"RDKit formal charge {formal_charge} differs from requested candidate charge {charge}."
        )
    return _Candidate(
        charge=formal_charge,
        mol=mol,
        smiles=smiles,
        explicit_h_smiles=explicit_h_smiles,
        inchi=inchi_value,
        inchikey=inchikey_value,
        formula=formula,
        score_tuple=score_tuple,
        score=score,
        warnings=warnings,
    )


def _perceive_candidates(
    fragment_block: str, charges: Sequence[int], config: PerceptionConfig
) -> tuple[list[_Candidate], list[str]]:
    candidates: list[_Candidate] = []
    errors: list[str] = []
    use_hueckel = config.use_hueckel
    if use_hueckel and not rdDetermineBonds.hueckelEnabled():
        use_hueckel = False
        errors.append("RDKit was not built with YAeHMOP; Hueckel mode disabled for this fragment.")

    for charge in charges:
        mol = _mol_from_xyz_block(fragment_block)
        try:
            mol = _determine_bonds(mol, int(charge), use_hueckel, config)
            if config.sanitize:
                Chem.SanitizeMol(mol)
            if mol.GetNumConformers():
                try:
                    Chem.AssignStereochemistryFrom3D(mol, confId=0, replaceExistingTags=True)
                except Exception:
                    pass
            candidates.append(_candidate_from_mol(mol, int(charge), config))
        except Exception as exc:
            errors.append(f"charge {charge}: {exc}")

    # Deduplicate exact charge+SMILES+InChI candidates, keeping the best-scored.
    best_by_key: dict[tuple[int, str, str | None], _Candidate] = {}
    for cand in candidates:
        key = (cand.charge, cand.smiles, cand.inchi)
        if key not in best_by_key or cand.score_tuple < best_by_key[key].score_tuple:
            best_by_key[key] = cand
    candidates = sorted(best_by_key.values(), key=lambda c: c.score_tuple)
    return candidates, errors


def _confidence(
    candidates: Sequence[_Candidate], chosen: _Candidate | None, charge_constrained: bool
) -> str:
    if chosen is None:
        return "none"
    unique = {(c.charge, c.smiles, c.inchi) for c in candidates}
    if charge_constrained and chosen.inchi is not None:
        return "high" if len(unique) == 1 else "medium"
    if len(unique) == 1 and chosen.inchi is not None:
        return "high"
    if len(candidates) > 1:
        second = candidates[1]
        if chosen.score_tuple[:3] == second.score_tuple[:3]:
            return "low"
    return "medium"


def _alternate_dict(cand: _Candidate) -> dict[str, Any]:
    return {
        "charge": cand.charge,
        "formula": cand.formula,
        "smiles": cand.smiles,
        "explicit_h_smiles": cand.explicit_h_smiles,
        "inchi": cand.inchi,
        "inchikey": cand.inchikey,
        "score": cand.score,
        "warnings": cand.warnings,
    }


def _candidate_to_record(
    cand: _Candidate | None,
    *,
    source: str | None,
    block_index: int,
    molecule_index: int,
    atom_indices: Sequence[int],
    elements: Sequence[str],
    candidates: Sequence[_Candidate],
    errors: Sequence[str],
    inherited_warnings: Sequence[str],
    charge_constrained: bool,
    config: PerceptionConfig,
) -> MoleculeRecord:
    if cand is None:
        return MoleculeRecord(
            source=source,
            block_index=block_index,
            molecule_index=molecule_index,
            atom_indices=list(atom_indices),
            elements=list(elements),
            formula=None,
            charge=None,
            smiles=None,
            explicit_h_smiles=None,
            inchi=None,
            inchikey=None,
            confidence="none",
            status="failed",
            warnings=list(inherited_warnings),
            errors=list(errors),
            alternates=[],
            mol=None,
        )
    alternates = [_alternate_dict(alt) for alt in candidates if alt is not cand][
        : config.max_reported_alternates
    ]
    warnings = list(inherited_warnings) + list(cand.warnings)
    if alternates:
        warnings.append(
            "Alternative charge/bond-order assignments were possible; inspect alternates."
        )
    return MoleculeRecord(
        source=source,
        block_index=block_index,
        molecule_index=molecule_index,
        atom_indices=list(atom_indices),
        elements=list(elements),
        formula=cand.formula,
        charge=cand.charge,
        smiles=cand.smiles,
        explicit_h_smiles=cand.explicit_h_smiles,
        inchi=cand.inchi,
        inchikey=cand.inchikey,
        confidence=_confidence(candidates, cand, charge_constrained),
        status="ok",
        score=dict(cand.score),
        warnings=warnings,
        errors=list(errors),
        alternates=alternates,
        mol=cand.mol,
    )


def _unsupported_elements(elements: Sequence[str], config: PerceptionConfig) -> list[str]:
    """Return the distinct elements RDKit cannot reliably perceive, in order."""
    if not config.restrict_to_supported_elements:
        return []
    allowed = config.supported_elements or DEFAULT_SUPPORTED_ELEMENTS
    unsupported: list[str] = []
    for element in elements:
        if element not in allowed and element not in unsupported:
            unsupported.append(element)
    return unsupported


def _unsupported_record(
    *,
    source: str | None,
    block_index: int,
    molecule_index: int,
    atom_indices: Sequence[int],
    elements: Sequence[str],
    unsupported: Sequence[str],
    inherited_warnings: Sequence[str],
) -> MoleculeRecord:
    reason = (
        "Contains element(s) RDKit bond perception cannot handle: "
        f"{', '.join(unsupported)}. xyz2mol / DetermineBonds targets main-group "
        "organics; metals and noble gases are not supported and would otherwise "
        "be mis-assigned (e.g. a lone Na+ comes back as [NaH] at high confidence). "
        "Set restrict_to_supported_elements=False to attempt perception anyway."
    )
    return MoleculeRecord(
        source=source,
        block_index=block_index,
        molecule_index=molecule_index,
        atom_indices=list(atom_indices),
        elements=list(elements),
        formula=None,
        charge=None,
        smiles=None,
        explicit_h_smiles=None,
        inchi=None,
        inchikey=None,
        confidence="none",
        status="unsupported",
        warnings=list(inherited_warnings),
        errors=[reason],
        alternates=[],
        mol=None,
    )


def _metal_ion_candidate(atom: _AtomRow, charge: int, config: PerceptionConfig) -> _Candidate:
    """Build a candidate for a single metal atom as its ion, bypassing RDKit perception."""
    rwmol = Chem.RWMol()
    rd_atom = Chem.Atom(atom.symbol)
    rd_atom.SetFormalCharge(int(charge))
    rd_atom.SetNoImplicit(True)
    rwmol.AddAtom(rd_atom)
    mol = rwmol.GetMol()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.SetAtomPosition(0, Point3D(atom.x, atom.y, atom.z))
    mol.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(mol)
    return _candidate_from_mol(mol, int(charge), config)


def _maybe_ionic_metal_candidate(
    atoms: Sequence[_AtomRow],
    atom_indices: Sequence[int],
    elements: Sequence[str],
    config: PerceptionConfig,
    n_fragments: int,
    frag_index: int,
) -> _Candidate | None:
    """Return an ion candidate for a lone group 1/2 metal counterion, else None."""
    if not config.recover_ionic_metals or len(atom_indices) != 1:
        return None
    oxidation_state = _ionic_metal_oxidation_state(elements[0])
    if oxidation_state is None:
        return None
    if config.per_fragment_charges is not None:
        charge = int(config.per_fragment_charges[frag_index])
    elif config.total_charge is not None and n_fragments == 1:
        charge = int(config.total_charge)
    else:
        charge = oxidation_state
    return _metal_ion_candidate(atoms[atom_indices[0]], charge, config)


def _with_ions(
    cand: _Candidate, ions: Sequence[_Candidate], config: PerceptionConfig
) -> _Candidate:
    """Merge metal-ion candidates into a candidate for the rest of the block."""
    mol = cand.mol
    for ion in ions:
        mol = Chem.CombineMols(mol, ion.mol)
    combined = _candidate_from_mol(mol, cand.charge + sum(ion.charge for ion in ions), config)
    combined.warnings[:0] = cand.warnings
    return combined


def _recover_whole_with_ionic_metals(
    atoms: Sequence[_AtomRow],
    config: PerceptionConfig,
    *,
    source: str | None,
    block_index: int,
    inherited_warnings: Sequence[str],
) -> list[MoleculeRecord] | None:
    """Whole-block perception of a block holding group 1/2 metal counterions.

    With ``split_fragments=False`` the block is a single component, so the ionic
    fast-path (which acts on lone-atom fragments) never sees a salt's metal, and
    the block was reported ``unsupported``. Here each group 1/2 metal atom is
    taken out as its ion, the remaining atoms are perceived as ONE component at
    ``total_charge`` minus the ions' charges, and the ions are merged back in,
    so the block still yields a single record with one SMILES and one InChIKey
    for the whole salt (e.g. ``[I-].[Na+]``).

    Returns ``None`` when this path does not apply: fragments are split,
    ``recover_ionic_metals`` is off, ``per_fragment_charges`` is given (the
    caller has fixed the component layout), or the block has no group 1/2 metal
    or nothing but metals.
    """
    if config.split_fragments or not config.recover_ionic_metals:
        return None
    if config.per_fragment_charges is not None:
        return None
    metal_idx = tuple(
        i for i, atom in enumerate(atoms) if _ionic_metal_oxidation_state(atom.symbol) is not None
    )
    rest_idx = tuple(
        i for i, atom in enumerate(atoms) if _ionic_metal_oxidation_state(atom.symbol) is None
    )
    if not metal_idx or not rest_idx:
        return None

    all_idx = rest_idx + metal_idx
    elements = [atoms[i].symbol for i in all_idx]
    ions = [
        _metal_ion_candidate(atoms[i], _ionic_metal_oxidation_state(atoms[i].symbol) or 0, config)
        for i in metal_idx
    ]
    ion_charge = sum(ion.charge for ion in ions)

    def failed(errors: Sequence[str]) -> list[MoleculeRecord]:
        return [
            _candidate_to_record(
                None,
                source=source,
                block_index=block_index,
                molecule_index=0,
                atom_indices=all_idx,
                elements=elements,
                candidates=[],
                errors=errors,
                inherited_warnings=inherited_warnings,
                charge_constrained=False,
                config=config,
            )
        ]

    rest_block, rest_elements = _fragment_xyz(
        atoms, rest_idx, comment=f"{source or 'xyz'} block={block_index} without ionic metals"
    )
    unsupported = _unsupported_elements(rest_elements, config)
    if unsupported:
        return [
            _unsupported_record(
                source=source,
                block_index=block_index,
                molecule_index=0,
                atom_indices=all_idx,
                elements=elements,
                unsupported=unsupported,
                inherited_warnings=inherited_warnings,
            )
        ]

    # A hydrogen whose only contact was the metal is free once the metal is
    # removed; reject it as the whole-block check does, or it comes back as [H+].
    if config.reject_isolated_hydrogens:
        isolated = _isolated_hydrogens(rest_block, [atoms[i] for i in rest_idx], config)
        if isolated:
            return failed(
                [
                    f"Hydrogen atom(s) {[rest_idx[k] for k in isolated]} have no bonded "
                    "neighbour other than a group 1/2 metal; bond-order assignment would "
                    "report them as free [H+]/[H-]. Rejected; try another conformer, or "
                    "set reject_isolated_hydrogens=False."
                ]
            )

    if config.total_charge is not None:
        charges: tuple[int, ...] = (int(config.total_charge) - ion_charge,)
    else:
        charges = _charge_search_order(config.candidate_charges)
    candidates, errors = _perceive_candidates(rest_block, charges, config)
    if not candidates and config.charge_fallback:
        extra_charges = tuple(
            c for c in _charge_search_order(config.candidate_charges) if c not in charges
        )
        if extra_charges:
            candidates, fb_errors = _perceive_candidates(rest_block, extra_charges, config)
            errors = list(errors) + fb_errors
            for cand in candidates:
                cand.warnings.append(
                    "Charge-constrained perception failed; recovered using "
                    "candidate_charges fallback."
                )
    if not candidates:
        return failed(errors)

    combined = [_with_ions(cand, ions, config) for cand in candidates]
    return [
        _candidate_to_record(
            combined[0],
            source=source,
            block_index=block_index,
            molecule_index=0,
            atom_indices=all_idx,
            elements=elements,
            candidates=combined,
            errors=errors,
            inherited_warnings=inherited_warnings,
            charge_constrained=config.total_charge is not None,
            config=config,
        )
    ]


def _combine_scores(
    a: tuple[int | float, ...], b: tuple[int | float, ...]
) -> tuple[int | float, ...]:
    return tuple(x + y for x, y in itertools.zip_longest(a, b, fillvalue=0))


def _choose_total_charge_combo(
    candidate_sets: Sequence[Sequence[_Candidate]], total_charge: int
) -> tuple[_Candidate, ...] | None:
    # Dynamic programming: for each running charge sum, keep the lowest-score combo.
    empty_score = (0, 0, 0, 0, 0, 0, 0.0)
    states: dict[int, tuple[tuple[_Candidate, ...], tuple[int | float, ...]]] = {
        0: ((), empty_score)
    }
    for cands in candidate_sets:
        next_states: dict[int, tuple[tuple[_Candidate, ...], tuple[int | float, ...]]] = {}
        for running_charge, (combo, score) in states.items():
            for cand in cands:
                new_charge = running_charge + cand.charge
                new_combo = combo + (cand,)
                new_score = _combine_scores(score, cand.score_tuple)
                if new_charge not in next_states or new_score < next_states[new_charge][1]:
                    next_states[new_charge] = (new_combo, new_score)
        states = next_states
    selected = states.get(int(total_charge))
    if selected is None:
        return None
    return selected[0]


def recover_xyz_block(
    block: str,
    config: PerceptionConfig | None = None,
    *,
    source: str | None = None,
    block_index: int = 0,
) -> list[MoleculeRecord]:
    """Recover molecule records from a single XYZ block."""
    config = config or PerceptionConfig()
    normal_blocks = parse_xyz_blocks(block)
    if len(normal_blocks) != 1:
        raise XyzRecoverError("recover_xyz_block expects exactly one XYZ block")
    normal_block = normal_blocks[0]
    atoms, _ = _atoms_from_xyz_block(normal_block)
    component_indices, split_warnings = _split_components(normal_block, atoms, config)
    n_fragments = len(component_indices)

    if config.per_fragment_charges is not None and len(config.per_fragment_charges) != n_fragments:
        raise XyzRecoverError(
            f"per_fragment_charges has {len(config.per_fragment_charges)} entries but the XYZ block has {n_fragments} disconnected components"
        )

    if config.reject_isolated_hydrogens:
        isolated = _isolated_hydrogens(normal_block, atoms, config)
        if isolated:
            reason = (
                f"Hydrogen atom(s) {isolated} have no bonded neighbour (an X-H bond "
                "stretched past the connectivity cutoff?); bond-order assignment would "
                "report them as free [H+]/[H-]. Rejected; try another conformer, or set "
                "reject_isolated_hydrogens=False."
            )
            return [
                _candidate_to_record(
                    None,
                    source=source,
                    block_index=block_index,
                    molecule_index=frag_idx,
                    atom_indices=atom_indices,
                    elements=[atoms[i].symbol for i in atom_indices],
                    candidates=[],
                    errors=[reason],
                    inherited_warnings=split_warnings,
                    charge_constrained=False,
                    config=config,
                )
                for frag_idx, atom_indices in enumerate(component_indices)
            ]

    whole_with_ions = _recover_whole_with_ionic_metals(
        atoms,
        config,
        source=source,
        block_index=block_index,
        inherited_warnings=split_warnings,
    )
    if whole_with_ions is not None:
        return whole_with_ions

    candidate_sets: list[list[_Candidate]] = []
    error_sets: list[list[str]] = []
    element_sets: list[list[str]] = []
    unsupported_sets: list[list[str]] = []
    fragment_blocks: list[str] = []
    for frag_idx, atom_indices in enumerate(component_indices):
        frag_block, elements = _fragment_xyz(
            atoms,
            atom_indices,
            comment=f"{source or 'xyz'} block={block_index} fragment={frag_idx}",
        )
        element_sets.append(elements)
        fragment_blocks.append(frag_block)

        ion_candidate = _maybe_ionic_metal_candidate(
            atoms, atom_indices, elements, config, n_fragments, frag_idx
        )
        if ion_candidate is not None:
            # Lone group 1/2 metal counterion: recover the ion directly.
            candidate_sets.append([ion_candidate])
            error_sets.append([])
            unsupported_sets.append([])
            continue

        unsupported = _unsupported_elements(elements, config)
        unsupported_sets.append(unsupported)
        if unsupported:
            # Skip perception entirely: RDKit would either raise or, worse,
            # return a confidently wrong structure for these elements.
            candidate_sets.append([])
            error_sets.append([])
            continue

        charges = _charges_for_fragment(config, n_fragments, frag_idx)
        candidates, errors = _perceive_candidates(frag_block, charges, config)
        if not candidates and config.charge_fallback:
            extra_charges = tuple(
                c for c in _charge_search_order(config.candidate_charges) if c not in charges
            )
            if extra_charges:
                candidates, fb_errors = _perceive_candidates(frag_block, extra_charges, config)
                errors = list(errors) + fb_errors
                for cand in candidates:
                    cand.warnings.append(
                        "Charge-constrained perception failed; recovered using "
                        "candidate_charges fallback."
                    )
        candidate_sets.append(candidates)
        error_sets.append(errors)

    chosen: list[_Candidate | None]
    charge_constrained = config.total_charge is not None or config.per_fragment_charges is not None
    if config.per_fragment_charges is not None:
        chosen = [cands[0] if cands else None for cands in candidate_sets]
    elif config.total_charge is not None:
        if any(not cands for cands in candidate_sets):
            chosen = [cands[0] if cands else None for cands in candidate_sets]
        else:
            combo = _choose_total_charge_combo(candidate_sets, config.total_charge)
            if combo is None:
                split_warnings = list(split_warnings) + [
                    f"No candidate combination matched total_charge={config.total_charge}; selected best unconstrained candidates."
                ]
                chosen = [cands[0] if cands else None for cands in candidate_sets]
                charge_constrained = False
            else:
                chosen = list(combo)
    else:
        chosen = [cands[0] if cands else None for cands in candidate_sets]

    records: list[MoleculeRecord] = []
    for frag_idx, atom_indices in enumerate(component_indices):
        if unsupported_sets[frag_idx]:
            records.append(
                _unsupported_record(
                    source=source,
                    block_index=block_index,
                    molecule_index=frag_idx,
                    atom_indices=atom_indices,
                    elements=element_sets[frag_idx],
                    unsupported=unsupported_sets[frag_idx],
                    inherited_warnings=split_warnings,
                )
            )
            continue
        records.append(
            _candidate_to_record(
                chosen[frag_idx],
                source=source,
                block_index=block_index,
                molecule_index=frag_idx,
                atom_indices=atom_indices,
                elements=element_sets[frag_idx],
                candidates=candidate_sets[frag_idx],
                errors=error_sets[frag_idx],
                inherited_warnings=split_warnings,
                charge_constrained=charge_constrained,
                config=config,
            )
        )
    return records


def recover_xyz_text(
    text: str,
    config: PerceptionConfig | None = None,
    *,
    source: str | None = None,
) -> list[MoleculeRecord]:
    """Recover molecule records from text containing one or more XYZ blocks.

    Each XYZ block is treated independently, so a multi-frame file (e.g. a
    trajectory of the same molecule) produces one set of records per frame,
    distinguished by the ``block_index`` field. De-duplicate on ``inchikey``
    downstream if you only want unique species.
    """
    config = config or PerceptionConfig()
    records: list[MoleculeRecord] = []
    for block_index, block in enumerate(parse_xyz_blocks(text)):
        records.extend(recover_xyz_block(block, config, source=source, block_index=block_index))
    return records


def recover_xyz_file(
    path: str | Path, config: PerceptionConfig | None = None
) -> list[MoleculeRecord]:
    """Recover molecule records from an XYZ file."""
    path = Path(path)
    return recover_xyz_text(path.read_text(), config or PerceptionConfig(), source=str(path))


def records_to_json(records: Sequence[MoleculeRecord], *, indent: int = 2) -> str:
    return json.dumps([record.to_dict() for record in records], indent=indent, sort_keys=True)


def write_sdf(records: Sequence[MoleculeRecord], path: str | Path) -> None:
    """Write successfully recovered molecules to SDF with result metadata."""
    writer = Chem.SDWriter(str(path))
    try:
        for record in records:
            if record.mol is None or record.status != "ok":
                continue
            mol = Chem.Mol(record.mol)
            for key, value in record.to_dict().items():
                if key == "mol":
                    continue
                if isinstance(value, (list, dict)):
                    mol.SetProp(key, json.dumps(value, sort_keys=True))
                elif value is not None:
                    mol.SetProp(key, str(value))
            writer.write(mol)
    finally:
        writer.close()


def write_csv(records: Sequence[MoleculeRecord], path: str | Path) -> None:
    """Write a flat CSV summary."""
    fields = [
        "source",
        "block_index",
        "molecule_index",
        "atom_indices",
        "formula",
        "charge",
        "smiles",
        "explicit_h_smiles",
        "inchi",
        "inchikey",
        "confidence",
        "status",
        "warnings",
        "errors",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = record.to_dict()
            row["atom_indices"] = json.dumps(row.get("atom_indices", []))
            row["warnings"] = json.dumps(row.get("warnings", []))
            row["errors"] = json.dumps(row.get("errors", []))
            writer.writerow({field: row.get(field) for field in fields})
