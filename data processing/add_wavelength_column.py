"""Add the wavelength column to Andor .asc files that were saved in pixel layout.

The PL GUIs save a spectrum image as ``Columns: wavelength_nm, intensity_row0..rowN`` (one line per
detector column: wavelength, then the rows).  Without a wavelength axis (spectrograph calibration
missing) the same image is saved as plain rows x columns of counts.  This rewrites such files in the
standard layout with the wavelength axis of a reference file taken at the same grating / center, so
viewers read them exactly like every other dataset.

    python add_wavelength_column.py <folder> --reference <file with wavelength column> [--out <folder>]

* Converted copies go to ``<folder>_wl`` (or ``--out``); the originals are never changed.
* Every original header line is kept; ``Columns: wavelength_nm, intensity_row0..rowN`` is added.
* Files whose grating / center wavelength / number of columns differ from the reference are refused.
* Files that already have a wavelength column, and every other file in the folder, are copied as they are.
* Files already in the output folder are skipped, so it can be run again when a scan has finished.
* Every converted file is read back and compared with the original counts and the reference axis.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

COLUMNS_LINE = "Columns: wavelength_nm, intensity_row0..rowN"


def read_header(path: Path):
    """Header lines (without the leading '# ') and the header as a {lower-case key: value} dict."""
    lines = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            text = line.rstrip("\r\n")
            lines.append(text[2:] if text.startswith("# ") else text[1:])
    meta = {}
    for text in lines:
        if ":" in text:
            k, v = text.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    return lines, meta


def has_wavelength_column(meta: dict) -> bool:
    return "wavelength" in meta.get("columns", "").lower()


def _same_setting(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return a == b


def load_reference(path: Path):
    lines, meta = read_header(path)
    if not has_wavelength_column(meta):
        raise SystemExit(f"Reference {path.name} has no wavelength column")
    data = np.loadtxt(path, comments="#", ndmin=2)
    return data[:, 0], meta


def save_standard(path: Path, header_lines, wl: np.ndarray, image: np.ndarray) -> None:
    """Exactly what AndorSystem.save_ascii writes for an image with a wavelength axis."""
    out = np.empty((image.shape[1], image.shape[0] + 1), dtype=np.float64)
    out[:, 0] = wl
    out[:, 1:] = image.T
    np.savetxt(str(path), out, fmt=["%g"] + ["%.10g"] * image.shape[0],
               header="\n".join(list(header_lines) + [COLUMNS_LINE]))


def convert_folder(src: Path, ref: Path, dst: Path) -> dict:
    wl_ref, ref_meta = load_reference(ref)
    dst.mkdir(parents=True, exist_ok=True)
    counts = {"converted": 0, "copied": 0, "skipped": 0, "refused": 0}
    refused = []
    for path in sorted(src.iterdir()):
        target = dst / path.name
        if path.is_dir():
            continue
        if path.suffix.lower() != ".asc":
            shutil.copy2(path, target)  # scan_timing.csv, session.md, ... (may change until the scan ends)
            counts["copied"] += 1
            continue
        if target.exists():
            counts["skipped"] += 1
            continue
        lines, meta = read_header(path)
        if has_wavelength_column(meta):
            shutil.copy2(path, target)
            counts["copied"] += 1
            continue
        why = [k for k in ("grating", "center_wl_nm") if not _same_setting(meta.get(k), ref_meta.get(k))]
        image = np.loadtxt(path, comments="#", ndmin=2)
        if image.shape[1] != wl_ref.size:
            why.append(f"{image.shape[1]} columns (reference {wl_ref.size})")
        if why:
            refused.append(f"{path.name}: differs from the reference in {', '.join(why)}")
            counts["refused"] += 1
            continue
        save_standard(target, lines, wl_ref, image)
        # read back: same counts, reference wavelengths, and the Columns line in the header
        back = np.loadtxt(target, comments="#", ndmin=2)
        _, back_meta = read_header(target)
        if not (has_wavelength_column(back_meta) and np.array_equal(back[:, 1:].T, image)
                and np.allclose(back[:, 0], wl_ref, rtol=0, atol=1e-9)):
            target.unlink()
            raise SystemExit(f"Read-back check failed for {path.name}; nothing kept for it")
        shutil.copystat(path, target)
        counts["converted"] += 1
    for line in refused:
        print("refused:", line)
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", type=Path, help="folder with pixel-layout .asc files")
    ap.add_argument("--reference", type=Path, required=True,
                    help=".asc file with a wavelength column, taken at the same grating / center wavelength")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <folder>_wl)")
    args = ap.parse_args(argv)
    src = args.folder.resolve()
    dst = (args.out or src.with_name(src.name + "_wl")).resolve()
    if dst == src:
        raise SystemExit("The output folder must differ from the input folder")
    _, ref_meta = read_header(args.reference)
    print(f"Reference: {args.reference.name} (grating {ref_meta.get('grating')}, "
          f"center {ref_meta.get('center_wl_nm')} nm)")
    counts = convert_folder(src, args.reference, dst)
    print(f"{dst}: {counts['converted']} converted, {counts['copied']} copied, "
          f"{counts['skipped']} already there, {counts['refused']} refused")
    return 1 if counts["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
