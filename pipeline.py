"""Automated ComfyUI + rembg + Alpha-scoring art pipeline for Van Kiem Quy Tong."""

from __future__ import annotations

import argparse
import gc
import io
import json
import secrets
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from alpha_analyzer import (
    analyze_alpha,
    build_contact_sheet,
    package_sprite,
    remove_background,
)
from assets_manifest import (
    ASSETS,
    ASSETS_BY_ID,
    CHECKPOINT_NAME,
    LORA_NAME,
    LORA_WEIGHT,
    AssetSpec,
)
from comfy_client import DEFAULT_COMFY_URL, ComfyClient

EXPORT_SUBFOLDERS = ("Characters", "Enemies", "Icons", "Environment", "UI_Kit")


def _ensure_dirs(root: Path) -> dict[str, Path]:
    export_root = root / "Art_Export"
    backup_root = root / "Backup_Candidates"
    for sub in EXPORT_SUBFOLDERS:
        (export_root / sub).mkdir(parents=True, exist_ok=True)
        (backup_root / sub).mkdir(parents=True, exist_ok=True)
    dirs = {
        "root": root,
        "export": export_root,
        "backup": backup_root,
        "raw": root / "Raw_Generated",
        "cutout": root / "Cutout_Candidates",
        "reports": root / "Reports",
    }
    for key in ("raw", "cutout", "reports"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def _generate_random_seeds(count: int, seed_base: int | None) -> list[int]:
    if seed_base is not None:
        return [int(seed_base) + i * 7919 for i in range(count)]
    seeds: set[int] = set()
    ordered: list[int] = []
    while len(ordered) < count:
        s = secrets.randbelow(2**32 - 1) + 1
        if s not in seeds:
            seeds.add(s)
            ordered.append(s)
    return ordered


def _load_existing_raw(raw_dir: Path) -> list[tuple[int, int, bytes]]:
    """Load existing vXX_seed_YYYY.png files ordered by variant index."""
    items: list[tuple[int, int, bytes]] = []
    for path in sorted(raw_dir.glob("v*_seed_*.png")):
        stem = path.stem
        try:
            prefix, seed_str = stem.split("_seed_", 1)
            v_idx = int(prefix.lstrip("v"))
            seed = int(seed_str)
        except ValueError:
            continue
        items.append((v_idx, seed, path.read_bytes()))
    items.sort(key=lambda x: x[0])
    return items


def _evaluate_items(
    items: list[tuple[int, int, bytes, float]],
    spec: AssetSpec,
    raw_dir: Path,
    cutout_dir: Path,
    use_cached_cutouts: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for v_idx, seed, png_bytes, gen_sec in items:
        cutout_file = cutout_dir / f"v{v_idx:02d}_seed_{seed}_cutout.png"
        t0 = time.monotonic()
        if use_cached_cutouts and cutout_file.is_file():
            with Image.open(cutout_file) as cached_im:
                cached_im.load()
                cutout = cached_im.convert("RGBA")
        else:
            with Image.open(io.BytesIO(png_bytes)) as raw_img:
                try:
                    cutout = remove_background(raw_img, spec)
                except PermissionError:
                    if not cutout_file.is_file():
                        raise
                    with Image.open(cutout_file) as cached_im:
                        cached_im.load()
                        cutout = cached_im.convert("RGBA")
            cutout.save(cutout_file, format="PNG")

        analysis = analyze_alpha(cutout, spec)
        proc_sec = round(time.monotonic() - t0, 2)

        status_tag = "PASS" if analysis.passed else "REJECT"
        reason_str = "" if analysis.passed else f" | reasons={list(analysis.reject_reasons)}"
        print(
            f"  [2/3] V{v_idx:02d} (seed={seed}): {status_tag} | "
            f"score={analysis.total_score:.2f} (center={analysis.center_score:.1f}, "
            f"clean={analysis.cleanliness_score:.1f}, frame={analysis.framing_score:.1f}) | "
            f"area={analysis.fg_area_ratio * 100:.1f}% frags={analysis.num_fragments} "
            f"deb={analysis.debris_ratio * 100:.1f}% margin={analysis.min_margin_px}px "
            f"[{proc_sec:.2f}s]{reason_str}",
            flush=True,
        )
        records.append(
            {
                "variant_idx": v_idx,
                "seed": seed,
                "gen_sec": gen_sec,
                "proc_sec": proc_sec,
                "raw_path": str(raw_dir / f"v{v_idx:02d}_seed_{seed}.png"),
                "cutout_path": str(cutout_file),
                "cutout": cutout,
                "analysis": analysis,
                "is_winner": False,
                "rank": None,
            }
        )
    return records


def process_asset(
    spec: AssetSpec,
    client: ComfyClient | None,
    dirs: dict[str, Path],
    variants: int = 8,
    steps: int = 35,
    cfg: float = 7.0,
    sampler_name: str = "euler_ancestral",
    scheduler: str = "normal",
    seed_base: int | None = None,
    force_regen: bool = False,
    reprocess_raw: bool = False,
    auto_retry: bool = True,
) -> dict[str, Any]:
    """Run Steps 1-3 (generate -> rembg + alpha filter -> rank & package) for one asset."""
    asset_started = time.monotonic()
    raw_dir = dirs["raw"] / spec.id
    cutout_dir = dirs["cutout"] / spec.id
    backup_sub_dir = dirs["backup"] / spec.subfolder
    raw_dir.mkdir(parents=True, exist_ok=True)
    cutout_dir.mkdir(parents=True, exist_ok=True)
    backup_sub_dir.mkdir(parents=True, exist_ok=True)

    export_path = dirs["export"] / spec.subfolder / f"{spec.id}.png"

    print(
        f"\n=== [{spec.id}] ({spec.subfolder}/{spec.id}.png | "
        f"target={spec.target_size[0]}x{spec.target_size[1]} | rembg={spec.remove_bg}) ===",
        flush=True,
    )

    raw_items: list[tuple[int, int, bytes, float]] = []
    if not force_regen:
        existing = _load_existing_raw(raw_dir)
        if existing:
            raw_items = [(v, s, b, 0.0) for v, s, b in existing[:variants]]
            print(f"  Resuming {len(raw_items)}/{variants} existing raw variant(s) from {raw_dir}", flush=True)

    if not reprocess_raw and len(raw_items) < variants:
        if client is None:
            raise RuntimeError("ComfyClient is required to generate missing variants.")
        existing_indices = {v for v, _, _, _ in raw_items}
        missing_indices = [i for i in range(1, variants + 1) if i not in existing_indices]
        new_seeds = _generate_random_seeds(len(missing_indices), seed_base)
        for v_idx, seed in zip(missing_indices, new_seeds):
            print(
                f"  [1/3] Generating variant {v_idx}/{variants} "
                f"(seed={seed}, steps={steps}, cfg={cfg}, size={spec.gen_size[0]}x{spec.gen_size[1]})...",
                flush=True,
            )
            gen = client.generate_variant(
                spec=spec,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
            )
            raw_file = raw_dir / f"v{v_idx:02d}_seed_{seed}.png"
            raw_file.write_bytes(gen.image_bytes)
            raw_items.append((v_idx, seed, gen.image_bytes, gen.elapsed_sec))
            print(f"        Done in {gen.elapsed_sec:.2f}s -> {raw_file.name}", flush=True)
        raw_items.sort(key=lambda x: x[0])

    if not raw_items:
        raise RuntimeError(f"No raw variants available for {spec.id}.")

    records = _evaluate_items(
        raw_items,
        spec,
        raw_dir,
        cutout_dir,
        use_cached_cutouts=reprocess_raw,
    )
    passing = [r for r in records if r["analysis"].passed]

    # Single automatic retry batch if every initial variant failed quality filters
    if not passing and auto_retry and not reprocess_raw and client is not None:
        print(
            f"  [!] 0/{len(records)} passed quality filters. Running 1 retry batch of {variants} variants...",
            flush=True,
        )
        retry_seeds = _generate_random_seeds(
            variants, None if seed_base is None else seed_base + 999983
        )
        retry_items: list[tuple[int, int, bytes, float]] = []
        base_idx = max(r["variant_idx"] for r in records)
        for offset, seed in enumerate(retry_seeds, start=1):
            v_idx = base_idx + offset
            print(f"  [Retry] Generating variant {v_idx} (seed={seed})...", flush=True)
            gen = client.generate_variant(
                spec=spec,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
            )
            raw_file = raw_dir / f"v{v_idx:02d}_seed_{seed}.png"
            raw_file.write_bytes(gen.image_bytes)
            retry_items.append((v_idx, seed, gen.image_bytes, gen.elapsed_sec))
        records.extend(_evaluate_items(retry_items, spec, raw_dir, cutout_dir))
        passing = [r for r in records if r["analysis"].passed]

    fallback_used = False
    ranked_pool = list(passing)
    if not ranked_pool:
        fallback_used = True
        non_clipped = [
            r
            for r in records
            if r["analysis"].fg_pixels > 0
            and not any(reason.startswith("clipped_at_edge") for reason in r["analysis"].reject_reasons)
        ]
        ranked_pool = non_clipped if non_clipped else [r for r in records if r["analysis"].fg_pixels > 0]
        if not ranked_pool:
            ranked_pool = list(records)
        print(
            f"  [!] Fallback activated for {spec.id}: selecting highest-scoring candidate from {len(ranked_pool)} available.",
            flush=True,
        )

    ranked_pool.sort(
        key=lambda r: (
            r["analysis"].total_score,
            r["analysis"].cleanliness_score,
            r["analysis"].center_score,
        ),
        reverse=True,
    )

    for rank, rec in enumerate(ranked_pool, start=1):
        rec["rank"] = rank
        if rank == 1:
            rec["is_winner"] = True

    # Remove stale backup files for this specific asset in Backup_Candidates/<subfolder>/
    for old_bk in backup_sub_dir.glob(f"{spec.id}_rank*.png"):
        old_bk.unlink()

    winner = ranked_pool[0]
    packaged_winner = package_sprite(winner["cutout"], spec)
    packaged_winner.save(export_path, format="PNG")
    winner_info = {
        "variant_idx": winner["variant_idx"],
        "seed": winner["seed"],
        "score": winner["analysis"].total_score,
        "fallback_used": fallback_used,
        "export_path": str(export_path),
    }
    print(
        f"  [3/3] EXPORT WINNER -> V{winner['variant_idx']:02d} (seed={winner['seed']}, "
        f"score={winner['analysis'].total_score:.2f}, fallback={fallback_used}) -> {export_path}",
        flush=True,
    )

    backup_paths: list[str] = []
    for rec in ranked_pool[1:]:
        packaged_bk = package_sprite(rec["cutout"], spec)
        bk_name = (
            f"{spec.id}_rank{rec['rank']}_v{rec['variant_idx']:02d}"
            f"_seed{rec['seed']}_score{rec['analysis'].total_score:.1f}.png"
        )
        bk_path = backup_sub_dir / bk_name
        packaged_bk.save(bk_path, format="PNG")
        backup_paths.append(str(bk_path))
    print(
        f"  [3/3] SAVED {len(backup_paths)} backup candidate(s) -> {backup_sub_dir}",
        flush=True,
    )

    contact_sheet_path = cutout_dir / f"{spec.id}_contact_sheet.png"
    build_contact_sheet(records, contact_sheet_path)

    report = {
        "asset_id": spec.id,
        "category": spec.category,
        "subfolder": spec.subfolder,
        "remove_bg": spec.remove_bg,
        "target_size": list(spec.target_size),
        "gen_size": list(spec.gen_size),
        "checkpoint": CHECKPOINT_NAME,
        "lora": LORA_NAME,
        "lora_weight": LORA_WEIGHT,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "total_variants": len(records),
        "passed_count": len(passing),
        "rejected_count": len(records) - len(passing),
        "fallback_used": fallback_used,
        "winner": winner_info,
        "backup_paths": backup_paths,
        "contact_sheet_path": str(contact_sheet_path),
        "elapsed_sec": round(time.monotonic() - asset_started, 2),
        "variants": [
            {
                "variant_idx": r["variant_idx"],
                "seed": r["seed"],
                "rank": r["rank"],
                "is_winner": r["is_winner"],
                "gen_sec": r["gen_sec"],
                "proc_sec": r["proc_sec"],
                "raw_path": r["raw_path"],
                "cutout_path": r["cutout_path"],
                "analysis": r["analysis"].to_dict(),
            }
            for r in records
        ],
    }

    report_path = dirs["reports"] / f"{spec.id}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Release PIL buffers between assets during long batch runs
    for rec in records:
        rec["cutout"] = None
    gc.collect()
    return report


def create_archive(
    root: Path,
    zip_name: str = "VanKiem_All_Art.zip",
    folders: tuple[str, ...] = ("Assets/Art", "Art_Export", "Backup_Candidates"),
) -> tuple[Path, int]:
    """Compress specified workspace folders into a verified zip archive at root."""
    zip_path = root / zip_name
    tmp_path = root / f".{zip_name}.tmp"
    file_count = 0

    with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for folder_name in folders:
            folder_path = root / folder_name
            if not folder_path.exists():
                continue
            for file_path in sorted(folder_path.rglob("*")):
                if file_path.is_file():
                    arcname = file_path.relative_to(root).as_posix()
                    zf.write(file_path, arcname=arcname)
                    file_count += 1

    with zipfile.ZipFile(tmp_path, mode="r") as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Zip integrity check failed at {bad_member}")

    if zip_path.exists():
        zip_path.unlink()
    tmp_path.replace(zip_path)
    return zip_path, file_count


def _resolve_targets(args: argparse.Namespace) -> list[AssetSpec]:
    if args.all:
        return list(ASSETS)
    if args.category:
        wanted = {c.lower() for c in args.category}
        selected = [s for s in ASSETS if s.category.lower() in wanted]
        if not selected:
            cats = sorted({s.category for s in ASSETS})
            raise SystemExit(f"No assets found for category {args.category}. Available: {cats}")
        return selected
    if args.asset:
        selected = []
        for asset_id in args.asset:
            if asset_id not in ASSETS_BY_ID:
                raise SystemExit(f"Unknown asset ID: '{asset_id}'. Run --list to inspect all IDs.")
            selected.append(ASSETS_BY_ID[asset_id])
        return selected
    raise SystemExit("Specify --asset <ID>, --category <CAT>, --all, --zip-only, or --list.")


def _print_manifest() -> None:
    print(f"Total assets in manifest: {len(ASSETS)}")
    for idx, s in enumerate(ASSETS, start=1):
        print(
            f"{idx:2d}. {s.subfolder}/{s.id:<22} | target={s.target_size[0]}x{s.target_size[1]} | "
            f"gen={s.gen_size[0]}x{s.gen_size[1]} | rembg={s.remove_bg}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated ComfyUI + rembg asset pipeline for Van Kiem Quy Tong."
    )
    parser.add_argument("--list", action="store_true", help="List all manifest assets and exit.")
    parser.add_argument("--asset", nargs="+", help="One or more asset IDs to process.")
    parser.add_argument("--category", nargs="+", help="One or more asset categories to process.")
    parser.add_argument("--all", action="store_true", help="Process all assets in manifest and zip.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip assets already in Art_Export/.")
    parser.add_argument("--force-regen", action="store_true", help="Ignore existing Raw_Generated PNGs.")
    parser.add_argument("--variants", type=int, default=8, help="Variants per asset (default: 8).")
    parser.add_argument("--steps", type=int, default=35, help="KSampler steps (default: 35).")
    parser.add_argument("--cfg", type=float, default=7.0, help="KSampler CFG scale (default: 7.0).")
    parser.add_argument("--sampler", default="euler_ancestral", help="ComfyUI sampler_name.")
    parser.add_argument("--scheduler", default="normal", help="ComfyUI scheduler.")
    parser.add_argument("--seed-base", type=int, default=None, help="Optional fixed seed base.")
    parser.add_argument("--comfy-url", default=DEFAULT_COMFY_URL, help="ComfyUI base URL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root output directory (defaults to script directory).",
    )
    parser.add_argument(
        "--reprocess-raw",
        action="store_true",
        help="Skip ComfyUI generation and reprocess existing PNGs in Raw_Generated/.",
    )
    parser.add_argument(
        "--no-auto-retry",
        action="store_true",
        help="Disable automatic retry batch when 0/8 candidates pass.",
    )
    parser.add_argument("--zip", action="store_true", help="Create VanKiem_All_Art.zip after run.")
    parser.add_argument("--zip-only", action="store_true", help="Only create VanKiem_All_Art.zip and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        _print_manifest()
        return 0

    root = args.output_dir.resolve()
    dirs = _ensure_dirs(root)

    if args.zip_only:
        zip_path, count = create_archive(root)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Created {zip_path} ({count} files, {size_mb:.2f} MB)")
        return 0

    targets = _resolve_targets(args)

    if args.skip_existing:
        filtered = []
        for spec in targets:
            out_file = dirs["export"] / spec.subfolder / f"{spec.id}.png"
            if out_file.exists():
                print(f"[SKIP] {spec.id} already exists at {out_file}")
            else:
                filtered.append(spec)
        targets = filtered

    client: ComfyClient | None = None
    if targets and not args.reprocess_raw:
        client = ComfyClient(base_url=args.comfy_url)
        stats = client.verify_server_and_models()
        device_name = stats.get("devices", [{}])[0].get("name", "unknown")
        print(
            f"Connected to ComfyUI at {args.comfy_url} | Device: {device_name} | "
            f"Checkpoint: {CHECKPOINT_NAME} | LoRA: {LORA_NAME} ({LORA_WEIGHT})",
            flush=True,
        )

    batch_reports: list[dict[str, Any]] = []
    batch_start = time.monotonic()
    for idx, spec in enumerate(targets, start=1):
        print(f"\n>>> Progress: [{idx}/{len(targets)}] {spec.subfolder}/{spec.id}", flush=True)
        rep = process_asset(
            spec=spec,
            client=client,
            dirs=dirs,
            variants=args.variants,
            steps=args.steps,
            cfg=args.cfg,
            sampler_name=args.sampler,
            scheduler=args.scheduler,
            seed_base=args.seed_base,
            force_regen=args.force_regen,
            reprocess_raw=args.reprocess_raw,
            auto_retry=not args.no_auto_retry,
        )
        batch_reports.append(rep)

    if batch_reports:
        summary_path = dirs["reports"] / "batch_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "total_assets_run": len(batch_reports),
                    "total_elapsed_sec": round(time.monotonic() - batch_start, 2),
                    "assets": [
                        {
                            "asset_id": r["asset_id"],
                            "subfolder": r["subfolder"],
                            "passed_count": r["passed_count"],
                            "total_variants": r["total_variants"],
                            "fallback_used": r["fallback_used"],
                            "winner": r["winner"],
                            "backup_count": len(r["backup_paths"]),
                            "elapsed_sec": r["elapsed_sec"],
                        }
                        for r in batch_reports
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nBatch complete ({len(batch_reports)} assets). Summary -> {summary_path}", flush=True)

    if args.all or args.zip:
        zip_path, count = create_archive(root)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Created archive: {zip_path} ({count} files, {size_mb:.2f} MB)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
