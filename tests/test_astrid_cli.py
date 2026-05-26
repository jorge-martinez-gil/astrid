from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from astrid_cli import cli_main
from audit_history import load_audit_runs


def test_cli_audit_can_save_history_to_custom_directory():
    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        data_path = tmp / "dataset.csv"
        history_dir = tmp / "history"
        pd.DataFrame(
            {
                "sample_id": ["a", "b", "c", "d"],
                "split": ["train", "train", "test", "test"],
                "target": [0, 1, 0, 1],
                "sensor": [1.0, 2.0, 1.5, 2.5],
            }
        ).to_csv(data_path, index=False)

        status = cli_main(
            [
                "audit",
                str(data_path),
                "--label",
                "target",
                "--split",
                "split",
                "--id",
                "sample_id",
                "--history-dir",
                str(history_dir),
                "--quiet",
            ]
        )
        runs = load_audit_runs(history_dir)

    assert status == 0
    assert len(runs) == 1
    assert runs[0]["analyzer"] == "tabular"
    assert runs[0]["dataset_name"] == "dataset.csv"


def test_cli_rejects_unknown_policy_name():
    try:
        cli_main(["audit", "missing.csv", "--policy", "Typo gate"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("CLI accepted an unknown policy name")


if __name__ == "__main__":
    test_cli_audit_can_save_history_to_custom_directory()
    test_cli_rejects_unknown_policy_name()
