# xyzrecover

`xyzrecover` is a small RDKit-based package for recovering molecular identifiers from XYZ coordinate files when the bond graph and charge were lost.

It is designed for batches of XYZ files that may contain one molecule, multiple disconnected covalent molecules, or multi-frame XYZ blocks. It uses RDKit's native `rdDetermineBonds` / `xyz2mol` implementation to infer connectivity and bond orders, then emits canonical SMILES, explicit-H SMILES, formal charge, formula, InChI, and InChIKey.

## Important limitation

XYZ files do not contain bond orders, formal charges, spin state, resonance choice, protonation intent, or non-covalent association metadata. For some geometries, more than one chemically valid assignment is possible. `xyzrecover` therefore searches candidate charges, scores the successful assignments, and includes alternates in the output instead of pretending the answer is always unique.

When you know the total charge, pass it. That is the single most useful constraint.

RDKit's valence model only handles main-group organic elements. Components
containing metals or noble gases are reported with `status="unsupported"` and
`smiles=None` rather than perceived, because RDKit would otherwise return a
silently wrong structure at high confidence (a lone Na⁺ comes back as `[NaH]`).
The supported set is `DEFAULT_SUPPORTED_ELEMENTS`; override it via
`PerceptionConfig(supported_elements=...)`, or disable the check with
`restrict_to_supported_elements=False` (CLI: `--no-element-check`).

The one exception is **ionic metal counterions**: an isolated group 1 / group 2
metal atom (Na, K, Li, Ca, Mg, …) is recovered directly as its ion (`[Na+]`,
`[Ca+2]`) using the group oxidation state, bypassing RDKit. Such atoms are split
out of any short ionic contact first, so a salt like Na⁺·methyl-phosphate⁻
recovers the metal as `[Na+]` and perceives the anion separately, and (when you
pass `total_charge`) the charge is balanced across the ion and the anion. This
is on by default; disable with `recover_ionic_metals=False` (CLI:
`--no-ionic-metals`). Transition metals and multi-atom metal fragments are still
reported `unsupported`.

## Install

```bash
python -m pip install rdkit
python -m pip install -e .
```

Conda also works well for RDKit:

```bash
conda install -c conda-forge rdkit
python -m pip install -e .
```

The package installs a `xyzrecover` console script; you can also run it as a
module with `python -m xyzrecover`.

> **Charge options apply to every block.** `--total-charge` and
> `--per-fragment-charges` are applied to *every* XYZ block in *every* input
> file. Use one charge state per invocation for heterogeneous batches.

## CLI examples

Unknown charge, try charge states -3 through +3:

```bash
xyzrecover molecule.xyz --json results.json --csv results.csv --sdf results.sdf
```

Known total charge for each XYZ block:

```bash
xyzrecover acetate.xyz --total-charge -1
```

A salt or cluster with multiple disconnected components and known total charge:

```bash
xyzrecover ion_pair.xyz --total-charge 0 --candidate-charges=-2,-1,0,1,2
```

Known per-component charges. The order follows connectivity perception (the
same order as `molecule_index` in the output, *not* the input atom order), so
run once unconstrained first to see how the structure splits:

```bash
xyzrecover fragments.xyz --per-fragment-charges=1,-1,0
```

If an asserted charge can't be satisfied for a component, fall back to the
candidate charge search for that component instead of failing it:

```bash
xyzrecover molecule.xyz --total-charge 2 --charge-fallback
```

Use extended Hueckel connectivity if your RDKit build supports YAeHMOP:

```bash
xyzrecover molecule.xyz --use-hueckel --total-charge 1
```

## Python API

```python
from xyzrecover import PerceptionConfig, recover_xyz_file, records_to_json

config = PerceptionConfig(
    total_charge=-1,                  # optional but recommended when known
    candidate_charges=(-2, -1, 0, 1), # used when fragment charge is unknown
    split_fragments=True,
)

records = recover_xyz_file("input.xyz", config)
for rec in records:
    print(rec.molecule_index, rec.formula, rec.charge, rec.smiles, rec.inchikey, rec.confidence)

print(records_to_json(records))
```

Each record contains:

- `status`: `ok`, `failed` (RDKit could not assign bonds), or `unsupported`
  (component contains an element RDKit cannot handle)
- `charge`: RDKit formal charge for the selected assignment
- `smiles`: canonical SMILES, with hydrogens made implicit when possible
- `explicit_h_smiles`: canonical SMILES retaining explicit hydrogens from the XYZ
- `inchi` and `inchikey`
- `confidence`: `high`, `medium`, `low`, or `none`
- `block_index` / `molecule_index`: which XYZ block (frame) and which recovered
  component the record came from
- `atom_indices`: indices into the source block's atoms for this component
- `alternates`: other successful charge/bond-order assignments, when present
- `warnings` / `errors`: notes and failed charge attempts, useful for debugging

### Multi-frame XYZ

Each XYZ block is processed independently, so a multi-frame file (e.g. a
trajectory of one molecule) yields one set of records per frame, distinguished
by `block_index`. De-duplicate on `inchikey` downstream if you only want unique
species.

## Recommended workflow

1. Run with known total charge whenever available.
2. Inspect any record with `confidence != "high"` or non-empty `alternates`.
3. For unusual bonding, radicals, metals, transition states, hypervalent species, or very close non-covalent contacts, validate against quantum chemistry output or another bond-perception method.
4. Keep the SDF output because it preserves the recovered 3D coordinates and the inferred bond graph.

## What to expect (benchmarks)

Recovery rate and confidence depend strongly on the chemistry. Measured across
several datasets (COMP6, ANI-1x/ANI-1xm, GDB, peptides, H-bonded and ion-pair
complexes):

| Regime | Perceived | Typical confidence | How to handle |
| --- | --- | --- | --- |
| Neutral organics (COMP6, ANI-1xm, GDB, peptides) | 98–100% | `high` | Auto-accept. Averaging over multiple conformers recovers strained cases (→99.8%). |
| Neutral H-bonded complexes | ~100% (matches OpenBabel) | `high` | Use `split_fragments=True` (default). |
| Proton-transfer / ion-pair complexes | recovered | `medium` | Genuinely ambiguous (where is the proton?); ≥ OpenBabel. Gate on confidence and inspect `alternates`. |
| Charged organic ions, charge unknown | ~86% | lower | Declare the charge (`--total-charge`) to push to `high`; otherwise gate. |
| Group 1/2 metal counterions (Na, K, Li, Ca, Mg) | lone ions recovered | `high` | Recovered as `[Na+]`/`[Ca+2]` via the ionic fast-path; pass `total_charge` to balance the anion. |
| Transition metals / multi-atom metal fragments | not supported | `none` | Reported as `status="unsupported"`, `smiles=None`. Use a metal-aware tool (e.g. xyz2mol_tm, xtb WBO). |

Practical gating rule: auto-accept `status="ok"` with `confidence="high"` and
empty `alternates`; route everything else (`medium`/`low`, non-empty
`alternates`, `failed`, `unsupported`) to review or a second method.

## Development

Install the dev extras (pytest, ruff, mypy) and run the checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest        # tests
ruff check .            # lint
ruff format .           # format
```

CI runs ruff (lint + format check) and pytest on Python 3.10–3.12.

## License

MIT — see [LICENSE](LICENSE).
