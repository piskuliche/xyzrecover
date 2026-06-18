from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import inchi as rd_inchi
from rdkit.Chem import rdDetermineBonds


class XyzRecoverError(RuntimeError):
    """Raised when XYZ parsing or molecule recovery cannot proceed."""


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
        `candidate_charges` for the corresponding components.
    candidate_charges
        Charge states to try when a component charge is unknown.
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
        data = asdict(self)
        if not include_mol:
            data.pop("mol", None)
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
                blocks.append(_format_xyz([a for a in atoms if a is not None], lines[i + 1] if i + 1 < len(lines) else ""))
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


def _split_components(block: str, config: PerceptionConfig) -> tuple[list[tuple[int, ...]], list[str]]:
    atoms, _ = _atoms_from_xyz_block(block)
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
            return frags, warnings
    except Exception as exc:  # pragma: no cover - fallback path depends on RDKit build/inputs
        warnings.append(f"RDKit connectivity failed; used covalent-radius fallback: {exc}")
    return _manual_components(atoms, config.cov_factor), warnings


def _fragment_xyz(block: str, indices: Sequence[int], comment: str = "") -> tuple[str, list[str]]:
    atoms, parent_comment = _atoms_from_xyz_block(block)
    selected = [atoms[i] for i in indices]
    frag_comment = comment or parent_comment or "fragment"
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


def _charges_for_fragment(config: PerceptionConfig, n_fragments: int, frag_index: int) -> tuple[int, ...]:
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


def _perceive_candidates(fragment_block: str, charges: Sequence[int], config: PerceptionConfig) -> tuple[list[_Candidate], list[str]]:
    candidates: list[_Candidate] = []
    errors: list[str] = []
    use_hueckel = config.use_hueckel
    if use_hueckel and not rdDetermineBonds.hueckelEnabled():
        use_hueckel = False
        errors.append("RDKit was not built with YAeHMOP; Hueckel mode disabled for this fragment.")

    for charge in charges:
        mol = _mol_from_xyz_block(fragment_block)
        try:
            rdDetermineBonds.DetermineBonds(
                mol,
                useHueckel=use_hueckel,
                charge=int(charge),
                covFactor=config.cov_factor,
                allowChargedFragments=config.allow_charged_fragments,
                embedChiral=config.embed_chiral,
                useAtomMap=False,
                useVdw=config.use_vdw,
                maxIterations=int(config.max_iterations),
            )
            if config.sanitize and not config.embed_chiral:
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


def _confidence(candidates: Sequence[_Candidate], chosen: _Candidate | None, charge_constrained: bool) -> str:
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
    alternates = [
        _alternate_dict(alt)
        for alt in candidates
        if alt is not cand
    ][: config.max_reported_alternates]
    warnings = list(inherited_warnings) + list(cand.warnings)
    if alternates:
        warnings.append("Alternative charge/bond-order assignments were possible; inspect alternates.")
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


def _combine_scores(a: tuple[int | float, ...], b: tuple[int | float, ...]) -> tuple[int | float, ...]:
    return tuple(x + y for x, y in itertools.zip_longest(a, b, fillvalue=0))


def _choose_total_charge_combo(
    candidate_sets: Sequence[Sequence[_Candidate]], total_charge: int
) -> tuple[_Candidate, ...] | None:
    # Dynamic programming: for each running charge sum, keep the lowest-score combo.
    empty_score = (0, 0, 0, 0, 0, 0, 0.0)
    states: dict[int, tuple[tuple[_Candidate, ...], tuple[int | float, ...]]] = {0: ((), empty_score)}
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
    component_indices, split_warnings = _split_components(normal_block, config)
    n_fragments = len(component_indices)

    if config.per_fragment_charges is not None and len(config.per_fragment_charges) != n_fragments:
        raise XyzRecoverError(
            f"per_fragment_charges has {len(config.per_fragment_charges)} entries but the XYZ block has {n_fragments} disconnected components"
        )

    candidate_sets: list[list[_Candidate]] = []
    error_sets: list[list[str]] = []
    element_sets: list[list[str]] = []
    fragment_blocks: list[str] = []
    for frag_idx, atom_indices in enumerate(component_indices):
        frag_block, elements = _fragment_xyz(
            normal_block,
            atom_indices,
            comment=f"{source or 'xyz'} block={block_index} fragment={frag_idx}",
        )
        charges = _charges_for_fragment(config, n_fragments, frag_idx)
        candidates, errors = _perceive_candidates(frag_block, charges, config)
        candidate_sets.append(candidates)
        error_sets.append(errors)
        element_sets.append(elements)
        fragment_blocks.append(frag_block)

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
    """Recover molecule records from text containing one or more XYZ blocks."""
    config = config or PerceptionConfig()
    records: list[MoleculeRecord] = []
    for block_index, block in enumerate(parse_xyz_blocks(text)):
        records.extend(recover_xyz_block(block, config, source=source, block_index=block_index))
    return records


def recover_xyz_file(path: str | Path, config: PerceptionConfig | None = None) -> list[MoleculeRecord]:
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
