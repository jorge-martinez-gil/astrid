# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jorge Martinez-Gil and the ASTRID authors. See LICENSE.
"""Command-line entry point for ASTRID.

Allows running tabular dataset audits headlessly — suitable for CI gates,
batch jobs, and scripts.

Example:
    astrid audit data.csv --preset Strict --json out.json
    astrid audit data.parquet --label target --split split --policy "Strict production"
    astrid audit data.csv --markdown report.md --html report.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from astrid_core import (
    TABULAR_PRESETS,
    analyze_tabular_file,
    build_tabular_recommendations,
    make_tabular_config,
    read_tabular_file,
)
from audit_history import POLICY_PRESETS, save_audit_record
from utils import build_html_report, build_markdown_report, to_json_safe


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="astrid",
        description="ASTRID — Advanced Software Tools for Reliable Industrial Datasets",
    )
    sub = p.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Run a tabular dataset audit")
    audit.add_argument("path", type=str, help="Path to a CSV, Parquet, XLS, or XLSX file")
    audit.add_argument("--preset", default="Balanced (recommended)",
                       choices=list(TABULAR_PRESETS.keys()), help="Threshold preset")
    audit.add_argument("--mode", default="Quick Scan",
                       choices=["Quick Scan", "Full Scan"], help="Scan mode")
    audit.add_argument("--policy", default=None, choices=list(POLICY_PRESETS.keys()),
                       help="Policy gate preset")
    audit.add_argument("--label", default=None, help="Label column name (optional)")
    audit.add_argument("--split", default=None, help="Split column name (optional)")
    audit.add_argument("--time", default=None, help="Time column name (optional)")
    audit.add_argument("--group", action="append", default=None,
                       help="Group column for fairness checks (repeatable)")
    audit.add_argument("--id", action="append", default=None, dest="ids",
                       help="ID column (repeatable)")
    audit.add_argument("--no-auto-columns", action="store_true",
                       help="Disable automatic column-role detection")
    audit.add_argument("--json", dest="json_out", default=None,
                       help="Write full JSON report to this path")
    audit.add_argument("--markdown", dest="md_out", default=None,
                       help="Write Markdown report to this path")
    audit.add_argument("--html", dest="html_out", default=None,
                       help="Write HTML report to this path")
    audit.add_argument("--save-history", action="store_true",
                       help="Save the audit record to the local ASTRID audit history")
    audit.add_argument("--history-dir", default=None,
                       help="Directory for saved audit history records; implies --save-history")
    audit.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress the human-readable summary on stdout")
    audit.add_argument("--exit-on-fail", action="store_true",
                       help="Exit with non-zero status if the policy gate fails")

    list_presets = sub.add_parser("presets", help="List available presets")
    list_presets.add_argument("kind", choices=["thresholds", "policy"], nargs="?",
                              default="thresholds")
    return p


def _cmd_presets(kind: str) -> int:
    if kind == "thresholds":
        for name, t in TABULAR_PRESETS.items():
            print(f"{name}:")
            print(f"  drift_ks_threshold     = {t.drift_ks_threshold}")
            print(f"  pii_hit_rate_threshold = {t.pii_hit_rate_threshold}")
    else:
        for name, p in POLICY_PRESETS.items():
            print(f"{name}:")
            for k, v in p.items():
                print(f"  {k:35s} = {v}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 2

    df, dataset_bytes = read_tabular_file(path)
    cfg = make_tabular_config(
        df,
        preset=args.preset,
        mode=args.mode,
        use_auto_columns=not args.no_auto_columns,
        label_col=args.label,
        split_col=args.split,
        time_col=args.time,
        group_cols=list(args.group) if args.group else None,
        id_cols=list(args.ids) if args.ids else None,
    )
    policy = POLICY_PRESETS.get(args.policy) if args.policy else None
    from astrid_core import analyze_tabular_dataframe
    result = analyze_tabular_dataframe(
        df,
        config=cfg,
        dataset_bytes=dataset_bytes,
        dataset_name=path.name,
        policy=policy,
    )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(to_json_safe(result), indent=2,
                                                  ensure_ascii=False), encoding="utf-8")

    cfg_dict_for_report = {
        **result.get("config", {}),
        "mode": cfg.mode,
        "preset": args.preset,
        "drift_ks_threshold": cfg.thresholds.drift_ks_threshold,
        "pii_hit_rate_threshold": cfg.thresholds.pii_hit_rate_threshold,
    }
    if args.md_out:
        md = build_markdown_report(
            df=df, report=result["report"], cfg_dict=cfg_dict_for_report,
            file_name=path.name, file_bytes=dataset_bytes,
            verdict=result["verdict"], reasons=result["findings"],
            recs=result["recommendations"], score=result["score"], grade=result["grade"],
        )
        Path(args.md_out).write_text(md, encoding="utf-8")
    if args.html_out:
        html = build_html_report(
            df=df, report=result["report"], cfg_dict=cfg_dict_for_report,
            file_name=path.name, file_bytes=dataset_bytes,
            verdict=result["verdict"], reasons=result["findings"],
            recs=result["recommendations"], score=result["score"], grade=result["grade"],
        )
        Path(args.html_out).write_text(html, encoding="utf-8")

    history_path = None
    if args.save_history or args.history_dir:
        try:
            history_path = save_audit_record(
                result["audit_record"],
                Path(args.history_dir) if args.history_dir else None,
            )
        except OSError as exc:
            print(f"Error: could not save audit history: {exc}", file=sys.stderr)
            return 2

    if not args.quiet:
        print(f"ASTRID audit: {path.name}")
        print(f"  Rows × Cols: {df.shape[0]:,} × {df.shape[1]:,}")
        print(f"  Score:       {result['score']}/100 (Grade {result['grade']})")
        print(f"  Verdict:     {result['verdict']}")
        if result.get("findings"):
            print("  Findings:")
            for f in result["findings"]:
                print(f"    - {f}")
        if args.policy:
            pol = result.get("policy_result", {})
            print(f"  Policy gate ({args.policy}): {pol.get('status', 'N/A')}")
        if history_path:
            print(f"  History:     {history_path}")

    if args.exit_on_fail:
        pol_status = (result.get("policy_result") or {}).get("status")
        if pol_status == "FAIL":
            return 1
    return 0


def cli_main(argv: Optional[list] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        return _cmd_audit(args)
    if args.command == "presets":
        return _cmd_presets(args.kind)
    return 1


if __name__ == "__main__":
    sys.exit(cli_main())
