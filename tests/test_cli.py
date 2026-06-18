import json
from pathlib import Path

from xyzrecover.cli import build_parser, main

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "water_methane.xyz"


def test_candidate_charges_default_is_none():
    # Default lives in PerceptionConfig; the CLI must not duplicate it.
    args = build_parser().parse_args([str(EXAMPLE)])
    assert args.candidate_charges is None


def test_cli_prints_json_to_stdout(capsys):
    rc = main([str(EXAMPLE)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert sorted(rec["smiles"] for rec in payload) == ["C", "O"]


def test_cli_writes_all_outputs(tmp_path, capsys):
    json_path = tmp_path / "r.json"
    csv_path = tmp_path / "r.csv"
    sdf_path = tmp_path / "r.sdf"
    rc = main(
        [
            str(EXAMPLE),
            "--total-charge",
            "0",
            "--json",
            str(json_path),
            "--csv",
            str(csv_path),
            "--sdf",
            str(sdf_path),
        ]
    )
    assert rc == 0
    assert json.loads(json_path.read_text())  # non-empty, valid JSON
    assert csv_path.read_text().startswith("source,")
    assert sdf_path.read_text().strip()  # SDF has content


def test_cli_missing_file_returns_error_code(capsys):
    rc = main(["does_not_exist.xyz"])
    assert rc == 2
    assert "failed to process" in capsys.readouterr().err
