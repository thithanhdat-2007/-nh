"""Final QA Audit, auto-remediation, and in-memory zip verification for Van Kiem Quy Tong."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from assets_manifest import ASSETS, AssetSpec
from comfy_client import DEFAULT_COMFY_URL, ComfyClient
from pipeline import EXPORT_SUBFOLDERS, _ensure_dirs, create_archive, process_asset


def audit_single_asset(
    spec: AssetSpec, export_root: Path
) -> tuple[bool, list[str], dict[str, Any]]:
    """Verify existence, file integrity, exact resolution, and Alpha transparency."""
    file_path = export_root / spec.subfolder / f"{spec.id}.png"
    issues: list[str] = []
    meta: dict[str, Any] = {
        "asset_id": spec.id,
        "subfolder": spec.subfolder,
        "path": str(file_path),
        "remove_bg": spec.remove_bg,
        "expected_size": list(spec.target_size),
        "actual_size": None,
        "file_bytes": 0,
        "mode": None,
        "transparent_ratio": None,
        "fg_ratio": None,
    }

    if not file_path.is_file():
        issues.append("missing_file")
        return False, issues, meta

    file_size = file_path.stat().st_size
    meta["file_bytes"] = file_size
    if file_size <= 0:
        issues.append("empty_file (0 bytes)")
        return False, issues, meta

    try:
        with Image.open(file_path) as im_verify:
            im_verify.verify()
        with Image.open(file_path) as im:
            im.load()
            actual_size = (im.width, im.height)
            meta["actual_size"] = list(actual_size)
            meta["mode"] = im.mode

            if actual_size != spec.target_size:
                issues.append(
                    f"size_mismatch (got {actual_size[0]}x{actual_size[1]}, "
                    f"expected {spec.target_size[0]}x{spec.target_size[1]})"
                )

            if spec.remove_bg:
                bands = im.getbands()
                if "A" not in bands:
                    issues.append(f"missing_alpha_channel (mode={im.mode})")
                else:
                    alpha = np.asarray(im.getchannel("A"), dtype=np.uint8)
                    total_px = float(alpha.size)
                    transparent_ratio = float(np.count_nonzero(alpha < 10)) / total_px
                    fg_ratio = float(np.count_nonzero(alpha > 20)) / total_px
                    meta["transparent_ratio"] = round(transparent_ratio, 4)
                    meta["fg_ratio"] = round(fg_ratio, 4)

                    if transparent_ratio < 0.05:
                        issues.append(
                            f"background_not_removed (transparent={transparent_ratio * 100:.2f}% < 5%)"
                        )
                    if fg_ratio < 0.02:
                        issues.append(
                            f"empty_foreground (fg={fg_ratio * 100:.2f}% < 2%)"
                        )
    except Exception as exc:
        issues.append(f"corrupt_image ({exc})")

    return len(issues) == 0, issues, meta


def verify_zip_in_memory(zip_path: Path) -> dict[str, Any]:
    """Verify every archive entry by reading bytes and verifying PNG headers in memory."""
    if not zip_path.is_file():
        raise RuntimeError(f"Zip archive missing: {zip_path}")

    checked_entries = 0
    checked_pngs = 0
    total_uncompressed = 0

    with zipfile.ZipFile(zip_path, mode="r") as zf:
        bad_crc = zf.testzip()
        if bad_crc is not None:
            raise RuntimeError(f"Zip CRC check failed at entry: {bad_crc}")

        for info in zf.infolist():
            if info.is_dir():
                continue
            data = zf.read(info.filename)
            if len(data) != info.file_size:
                raise RuntimeError(
                    f"Size mismatch reading {info.filename}: got {len(data)}, expected {info.file_size}"
                )
            total_uncompressed += len(data)
            checked_entries += 1
            if info.filename.lower().endswith(".png"):
                with Image.open(io.BytesIO(data)) as im:
                    im.verify()
                checked_pngs += 1

    return {
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_mb": round(zip_path.stat().st_size / (1024 * 1024), 2),
        "checked_entries": checked_entries,
        "checked_pngs": checked_pngs,
        "total_uncompressed_mb": round(total_uncompressed / (1024 * 1024), 2),
    }


def rebuild_consolidated_batch_summary(reports_dir: Path) -> dict[str, Any]:
    """Rebuild Reports/batch_summary.json across all manifest assets from per-asset reports."""
    entries: list[dict[str, Any]] = []
    for spec in ASSETS:
        rep_path = reports_dir / f"{spec.id}_report.json"
        if not rep_path.is_file():
            continue
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "asset_id": rep["asset_id"],
                "subfolder": rep["subfolder"],
                "passed_count": rep["passed_count"],
                "total_variants": rep["total_variants"],
                "fallback_used": rep.get("fallback_used", False),
                "winner": rep.get("winner"),
                "backup_count": len(rep.get("backup_paths", [])),
                "elapsed_sec": rep.get("elapsed_sec", 0.0),
            }
        )

    summary = {
        "total_assets_manifest": len(ASSETS),
        "total_assets_reported": len(entries),
        "assets": entries,
    }
    (reports_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def run_qa_audit(
    root: Path,
    auto_fix: bool = True,
    max_retries: int = 3,
    comfy_url: str = DEFAULT_COMFY_URL,
) -> int:
    """Audit all manifest assets, auto-remediate any failures, and verify zip archive."""
    dirs = _ensure_dirs(root)
    export_root = dirs["export"]
    backup_root = dirs["backup"]
    reports_dir = dirs["reports"]

    print(f"=== FINAL QA AUDIT ({len(ASSETS)} manifest assets) ===", flush=True)
    results: list[dict[str, Any]] = []
    failed_specs: list[tuple[AssetSpec, list[str]]] = []

    for spec in ASSETS:
        ok, issues, meta = audit_single_asset(spec, export_root)
        meta["passed"] = ok
        meta["issues"] = issues
        meta["remediated"] = False
        results.append(meta)
        if not ok:
            failed_specs.append((spec, issues))

    print(
        f"Initial audit: {len(ASSETS) - len(failed_specs)}/{len(ASSETS)} passed.",
        flush=True,
    )
    for spec, issues in failed_specs:
        print(f"  [FAIL] {spec.subfolder}/{spec.id}: {issues}", flush=True)

    if failed_specs and not auto_fix:
        return 1

    if failed_specs and auto_fix:
        client = ComfyClient(base_url=comfy_url)
        client.verify_server_and_models()
        results_by_id = {r["asset_id"]: r for r in results}

        for spec, initial_issues in failed_specs:
            print(
                f"\n[AUTO-FIX] Remediating {spec.subfolder}/{spec.id} (issues={initial_issues})...",
                flush=True,
            )
            fixed = False
            last_issues = initial_issues
            for attempt in range(1, max_retries + 1):
                print(
                    f"  [Attempt {attempt}/{max_retries}] Running pipeline for {spec.id}...",
                    flush=True,
                )
                process_asset(
                    spec=spec,
                    client=client,
                    dirs=dirs,
                    force_regen=(attempt > 1 or "missing_file" not in last_issues),
                )
                ok, issues, meta = audit_single_asset(spec, export_root)
                meta["passed"] = ok
                meta["issues"] = issues
                meta["remediated"] = True
                results_by_id[spec.id] = meta
                if ok:
                    print(f"  [FIXED] {spec.id} passed QA on attempt {attempt}.", flush=True)
                    fixed = True
                    break
                last_issues = issues
                print(f"  [STILL FAILING] {spec.id}: {issues}", flush=True)

            if not fixed:
                print(
                    f"ERROR: {spec.id} still failed QA after {max_retries} attempts: {last_issues}",
                    flush=True,
                )
                return 2

        results = [results_by_id[spec.id] for spec in ASSETS]

    # Count backups per subfolder
    backup_counts: dict[str, int] = defaultdict(int)
    for sub in EXPORT_SUBFOLDERS:
        sub_dir = backup_root / sub
        if sub_dir.is_dir():
            backup_counts[sub] = sum(1 for p in sub_dir.glob("*.png") if p.is_file())

    export_counts: dict[str, int] = defaultdict(int)
    pass_counts: dict[str, int] = defaultdict(int)
    for meta in results:
        sub = meta["subfolder"]
        if meta["file_bytes"] > 0:
            export_counts[sub] += 1
        if meta["passed"]:
            pass_counts[sub] += 1

    total_passed = sum(1 for r in results if r["passed"])
    total_backups = sum(backup_counts.values())

    zip_info: dict[str, Any] | None = None
    if auto_fix:
        rebuild_consolidated_batch_summary(reports_dir)
        zip_path, archived_count = create_archive(root, "VanKiem_All_Art.zip")
        zip_info = verify_zip_in_memory(zip_path)
        zip_info["archived_count"] = archived_count

    qa_report = {
        "audited_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_manifest": len(ASSETS),
        "total_passed": total_passed,
        "total_backups": total_backups,
        "by_subfolder": {
            sub: {
                "exported": export_counts[sub],
                "passed": pass_counts[sub],
                "backups": backup_counts[sub],
            }
            for sub in EXPORT_SUBFOLDERS
        },
        "zip_verification": zip_info,
        "assets": results,
    }
    (reports_dir / "qa_audit_report.json").write_text(
        json.dumps(qa_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n================ FINAL QA AUDIT SUMMARY ================", flush=True)
    print(f"{'Subfolder':<14} | {'Exported':>8} | {'QA Pass':>8} | {'Backups':>8}")
    print("-" * 48)
    for sub in EXPORT_SUBFOLDERS:
        print(
            f"{sub:<14} | {export_counts[sub]:>8} | {pass_counts[sub]:>8} | {backup_counts[sub]:>8}"
        )
    print("-" * 48)
    print(f"{'TOTAL':<14} | {sum(export_counts.values()):>8} | {total_passed:>5}/{len(ASSETS)} | {total_backups:>8}")

    if zip_info is not None:
        print(
            f"\nArchive: {zip_info['zip_path']} | Size: {zip_info['zip_mb']:.2f} MB "
            f"({zip_info['zip_bytes']} bytes) | Entries verified in memory: "
            f"{zip_info['checked_entries']} ({zip_info['checked_pngs']} PNGs)"
        )
    if total_passed == len(ASSETS) and (zip_info is not None or not auto_fix):
        print("\n>>> SẴN SÀNG CHUYỂN SANG MÁY CŨ <<<", flush=True)
    return 0 if total_passed == len(ASSETS) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final QA Audit for Van Kiem Quy Tong assets.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check existing Art_Export files without auto-fixing or re-zipping.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Max remediation attempts per asset.")
    parser.add_argument("--comfy-url", default=DEFAULT_COMFY_URL, help="ComfyUI server URL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root output directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_qa_audit(
        root=args.output_dir.resolve(),
        auto_fix=not args.check_only,
        max_retries=args.max_retries,
        comfy_url=args.comfy_url,
    )


if __name__ == "__main__":
    sys.exit(main())
