from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import PerceptionConfig, records_to_json, recover_xyz_file, write_csv, write_sdf


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value.strip():
        return tuple()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xyzrecover",
        description="Recover SMILES, charge, InChI, and InChIKey from XYZ files using RDKit.",
    )
    parser.add_argument("xyz", nargs="+", help="XYZ file(s) to process")
    parser.add_argument(
        "--total-charge",
        type=int,
        default=None,
        help="Known total charge. Applied to EVERY XYZ block in EVERY input file, "
        "so use one charge state per invocation for heterogeneous inputs.",
    )
    parser.add_argument(
        "--per-fragment-charges",
        type=_parse_int_tuple,
        default=None,
        help="Comma-separated charges for disconnected components, e.g. '1,-1,0'. "
        "Order follows connectivity perception (the molecule_index order in the "
        "output), and is applied to every block, so every block must split into "
        "the same number of components.",
    )
    parser.add_argument(
        "--candidate-charges",
        type=_parse_int_tuple,
        default=None,
        help="Comma-separated charge states to try when charge is unknown (default: -3..3)",
    )
    parser.add_argument(
        "--charge-fallback",
        action="store_true",
        help="If an asserted charge (--total-charge / --per-fragment-charges) cannot "
        "be satisfied for a component, fall back to --candidate-charges for it",
    )
    parser.add_argument(
        "--no-split", action="store_true", help="Do not split disconnected covalent components"
    )
    parser.add_argument(
        "--use-hueckel",
        action="store_true",
        help="Use extended Hueckel connectivity if RDKit supports it",
    )
    parser.add_argument(
        "--use-vdw", action="store_true", help="Use RDKit van der Waals connectivity"
    )
    parser.add_argument(
        "--cov-factor", type=float, default=1.30, help="Covalent-radius multiplier for connectivity"
    )
    parser.add_argument(
        "--radicals-instead-of-charges",
        action="store_true",
        help="Tell RDKit to use radicals instead of charged fragments when assigning valence",
    )
    parser.add_argument(
        "--no-chiral", action="store_true", help="Do not embed 3D-derived chirality"
    )
    parser.add_argument(
        "--no-sanitize", action="store_true", help="Do not run RDKit sanitization on candidates"
    )
    parser.add_argument(
        "--no-element-check",
        action="store_true",
        help="Attempt perception even for elements RDKit cannot handle (metals, "
        "noble gases); by default these are reported as status='unsupported'",
    )
    parser.add_argument(
        "--no-ionic-metals",
        action="store_true",
        help="Do not recover lone group 1/2 metal counterions as their ion "
        "([Na+], [Ca+2], ...); report them as status='unsupported' instead",
    )
    parser.add_argument(
        "--keep-explicit-h-smiles",
        action="store_true",
        help="Use explicit-H SMILES as the primary SMILES",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="RDKit DetermineBonds maxIterations; 0 means no cap",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write JSON output to this path")
    parser.add_argument("--csv", type=Path, default=None, help="Write CSV summary to this path")
    parser.add_argument("--sdf", type=Path, default=None, help="Write recovered molecules to SDF")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_kwargs: dict = dict(
        total_charge=args.total_charge,
        per_fragment_charges=args.per_fragment_charges,
        charge_fallback=args.charge_fallback,
        split_fragments=not args.no_split,
        allow_charged_fragments=not args.radicals_instead_of_charges,
        use_hueckel=args.use_hueckel,
        use_vdw=args.use_vdw,
        cov_factor=args.cov_factor,
        embed_chiral=not args.no_chiral,
        sanitize=not args.no_sanitize,
        restrict_to_supported_elements=not args.no_element_check,
        recover_ionic_metals=not args.no_ionic_metals,
        keep_explicit_h_smiles=args.keep_explicit_h_smiles,
        max_iterations=args.max_iterations,
    )
    # Only override the config default when the user actually passed the flag,
    # so the default charge set lives in exactly one place (PerceptionConfig).
    if args.candidate_charges is not None:
        config_kwargs["candidate_charges"] = args.candidate_charges
    config = PerceptionConfig(**config_kwargs)

    records = []
    exit_code = 0
    for xyz_path in args.xyz:
        try:
            records.extend(recover_xyz_file(xyz_path, config))
        except Exception as exc:
            exit_code = 2
            print(f"xyzrecover: failed to process {xyz_path}: {exc}", file=sys.stderr)

    output = records_to_json(records)
    if args.json:
        args.json.write_text(output + "\n")
    else:
        print(output)
    if args.csv:
        write_csv(records, args.csv)
    if args.sdf:
        write_sdf(records, args.sdf)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
