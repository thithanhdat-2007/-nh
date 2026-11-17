"""Deep 9x9 Alpha audit and enclosed solid-background cleanup for Art_Export/."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SKIP_SOLID_BG = frozenset(
    {
        "Env_ArenaFloor.png",
        "UI_MainMenu_BG.png",
        "App_Icon.png",
        "UI_Bar_Fill_White.png",
    }
)


def _ensure_onnx_dll_path() -> None:
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        scripts_dir = Path(sys.prefix) / "Scripts"
        if scripts_dir.is_dir():
            os.add_dll_directory(str(scripts_dir))


def _is_flagged(arr: np.ndarray, filename: str) -> bool:
    """Flag files with >15% opaque black/white on a 9x9 grid or opaque black/white UI center."""
    h, w, _ = arr.shape
    ys = np.linspace(0, h - 1, 9, dtype=int)
    xs = np.linspace(0, w - 1, 9, dtype=int)
    grid = arr[np.ix_(ys, xs)]
    r, g, b, a = grid[:, :, 0], grid[:, :, 1], grid[:, :, 2], grid[:, :, 3]
    opaque = a > 220
    black = opaque & (r < 25) & (g < 25) & (b < 25)
    white = opaque & (r > 235) & (g > 235) & (b > 235)
    if (black | white).sum() / 81.0 > 0.15:
        return True

    if any(k in filename for k in ("UI", "Border", "Frame")):
        cp = arr[h // 2, w // 2]
        if cp[3] > 220 and (
            (cp[0] < 25 and cp[1] < 25 and cp[2] < 25)
            or (cp[0] > 235 and cp[1] > 235 and cp[2] > 235)
        ):
            return True
    return False


def _clear_component_softly(out: np.ndarray, comp: np.ndarray) -> None:
    """Clear an enclosed background component while preserving glowing/ink edges."""
    core = ndimage.binary_erosion(comp, iterations=1)
    rim = comp & ~core
    out[core, 3] = 0
    out[rim, 3] = np.minimum(out[rim, 3], 64)


def _cleanup_enclosed_pockets(arr: np.ndarray, rel_path: str) -> np.ndarray:
    """Remove enclosed opaque black/white backgrounds trapped inside frames/seals."""
    out = arr.copy()
    h, w, _ = out.shape
    tot = float(h * w)
    name = Path(rel_path).name
    r, g, b, a = out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

    # 1) UI/Border/Frame with solid black or white center trapped inside frame
    if any(k in name for k in ("UI", "Border", "Frame")):
        cy, cx = h // 2, w // 2
        if a[cy, cx] > 200:
            for mask in (
                (a > 200) & (r < 25) & (g < 25) & (b < 25),
                (a > 200) & (r > 235) & (g > 235) & (b > 235),
            ):
                if not mask[cy, cx]:
                    continue
                lbl, _ = ndimage.label(mask)
                comp = lbl == lbl[cy, cx]
                touches_edge = (
                    comp[0, :].any()
                    or comp[-1, :].any()
                    or comp[:, 0].any()
                    or comp[:, -1].any()
                )
                if not touches_edge and comp.sum() > 0.03 * tot:
                    _clear_component_softly(out, comp)
                    break

    # 2) Circular seal badges (Tag_*) and enclosed pure-white UI icon pockets (UI_Icon_*)
    #    Characters/Enemies intentionally preserve white robes via alpha_analyzer.py.
    if name.startswith("Tag_") or name.startswith("UI_Icon_"):
        r, g, b, a = out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]
        pure_white = (a > 200) & (r > 235) & (g > 235) & (b > 235)
        lbl, n_comp = ndimage.label(pure_white)
        for cid in range(1, n_comp + 1):
            comp = lbl == cid
            touches_edge = (
                comp[0, :].any()
                or comp[-1, :].any()
                or comp[:, 0].any()
                or comp[:, -1].any()
            )
            if not touches_edge and comp.sum() > 0.03 * tot:
                _clear_component_softly(out, comp)

    return out


def _cleanup_asset(
    arr: np.ndarray,
    rel_path: str,
    rembg_mod: object,
    rembg_session: object,
) -> tuple[np.ndarray, float]:
    """Run rembg on white-composited RGB + enclosed pocket cleanup without harming glow."""
    h, w, _ = arr.shape
    name = Path(rel_path).name
    cleaned = arr.copy()

    # For Character/Enemy sprites, skip aggressive white-robe removal by bria-rmbg
    is_char_or_enemy = rel_path.startswith(("Characters/", "Enemies/"))
    if not is_char_or_enemy:
        rgba_im = Image.fromarray(arr, mode="RGBA")
        white_bg = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        rgb = Image.alpha_composite(white_bg, rgba_im).convert("RGB")
        bria_arr = np.asarray(
            rembg_mod.remove(rgb, session=rembg_session).convert("RGBA")
        )
        # Only lower opaque pixels (alpha > 200); preserve semi-transparent glow (alpha <= 200)
        opaque_mask = arr[:, :, 3] > 200
        cleaned[opaque_mask, 3] = np.minimum(
            arr[opaque_mask, 3], bria_arr[opaque_mask, 3]
        )

    cleaned = _cleanup_enclosed_pockets(cleaned, rel_path)
    newly_cleared = ((arr[:, :, 3] > 200) & (cleaned[:, :, 3] < 20)).mean()
    return cleaned, float(newly_cleared)


def sync_to_assets_art(export_root: Path, assets_art_root: Path) -> None:
    """Mirror Art_Export/ PNGs into Assets/Art/."""
    for png_path in sorted(export_root.rglob("*.png")):
        rel = png_path.relative_to(export_root)
        dst = assets_art_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png_path, dst)


def rebuild_export_zip(export_root: Path, zip_path: Path) -> None:
    """Rebuild VanKiem_Art_Export.zip atomically."""
    tmp_path = zip_path.with_name(f".{zip_path.name}.tmp")
    with zipfile.ZipFile(
        tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for file_path in sorted(export_root.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.as_posix())
    tmp_path.replace(zip_path)


def run_audit(dry_run: bool = False) -> list[str]:
    export_root = Path("Art_Export")
    assets_art_root = Path("Assets/Art")
    zip_path = Path("VanKiem_Art_Export.zip")

    rembg_mod = None
    rembg_session = None
    reprocessed: list[str] = []

    for png_path in sorted(export_root.rglob("*.png")):
        if png_path.name in SKIP_SOLID_BG:
            continue

        with Image.open(png_path) as im:
            arr = np.asarray(im.convert("RGBA"))

        if not _is_flagged(arr, png_path.name):
            continue

        if rembg_mod is None:
            _ensure_onnx_dll_path()
            import rembg as _rembg

            rembg_mod = _rembg
            rembg_session = rembg_mod.new_session(providers=["CPUExecutionProvider"])

        rel = png_path.relative_to(export_root).as_posix()
        cleaned, cleared_ratio = _cleanup_asset(arr, rel, rembg_mod, rembg_session)

        # Require >= 1% of pixels newly cleared so clean Sumi-e assets aren't needlessly rewritten
        if cleared_ratio < 0.01:
            continue

        reprocessed.append(rel)
        if dry_run:
            old_tr = (arr[:, :, 3] < 10).mean() * 100
            new_tr = (cleaned[:, :, 3] < 10).mean() * 100
            print(f"[DRY-RUN] {rel} | transp: {old_tr:.1f}% -> {new_tr:.1f}% | cleared={cleared_ratio*100:.1f}%")
        else:
            Image.fromarray(cleaned, mode="RGBA").save(png_path, format="PNG")

    if not dry_run:
        sync_to_assets_art(export_root, assets_art_root)
        rebuild_export_zip(export_root, zip_path)
        for rel in reprocessed:
            print(rel)

    return reprocessed


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep 9x9 Alpha audit for Art_Export.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without writing files.")
    args = parser.parse_args()
    run_audit(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
