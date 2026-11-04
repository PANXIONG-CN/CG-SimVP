"""Generate the declared QD/MLT cache from user-supplied TEC coordinates.

This public-release implementation reconstructs the documented cache interface.
The historical generator was a separate author script.
"""
import argparse
import calendar
import datetime
import importlib.metadata
import json
from pathlib import Path
import tempfile

import numpy as np

from scenario_core import digest, load_grid
from solar_zenith import FEATURE_VERSION, configure_geometry


APEXPY_VERSION = "2.1.1"
SUBSOLAR_HEIGHT_KM = 318550.0
EPOCH_POLICY = "Fixed July-1 IGRF epoch within each calendar year"


def prepare_grid(grid_path, latitude_file, longitude_file, source_description):
    """Preserve coordinate order and attach the user's source description."""
    grid_path = Path(grid_path)
    if bool(latitude_file) != bool(longitude_file):
        raise ValueError("Supply both --latitude-file and --longitude-file")
    sources = {}
    if latitude_file and longitude_file:
        latitude_file, longitude_file = Path(latitude_file), Path(longitude_file)
        lat = np.load(latitude_file, allow_pickle=False)
        lon = np.load(longitude_file, allow_pickle=False)
        sources = {
            "latitude_file_sha256": digest(latitude_file),
            "longitude_file_sha256": digest(longitude_file),
        }
        if grid_path.exists():
            old_lat, old_lon, _ = load_grid(grid_path)
            if not np.array_equal(lat, old_lat) or not np.array_equal(lon, old_lon):
                raise FileExistsError("Existing grid differs from coordinate arrays. Choose a new --grid-npz.")
        else:
            grid_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=grid_path.parent) as directory:
                temporary = Path(directory)/"grid.npz"
                np.savez(temporary, lat=lat, lon=lon)
                load_grid(temporary)
                temporary.rename(grid_path)
    lat, lon, _ = load_grid(grid_path)
    provenance = {
        "source_description": source_description,
        "coordinate_order": "Copied from supplied arrays without sorting or flipping",
        "grid_sha256": digest(grid_path),
        "shape": [len(lat), len(lon)],
        "latitude_units": "geodetic degrees north",
        "longitude_units": "degrees east",
        "input_coordinate_hashes": sources,
        "verification_scope": "Shape, finite coordinates, monotonic latitude, cyclic longitude coverage",
        "required_author_check": "Coordinate order and geodetic convention must match the TEC preprocessing",
    }
    provenance_path = grid_path.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance_path


def prepare_year(grid_path, cache_root, year, height_km=450.0, frames_per_day=96):
    """Write qd_lat.npy, qd_lon.npy, mlt0_hours.npy and metadata.json."""
    import apexpy

    version = importlib.metadata.version("apexpy")
    if version != APEXPY_VERSION:
        raise ValueError(f"This release requires apexpy=={APEXPY_VERSION}, found {version}")
    if frames_per_day <= 0 or 86400 % frames_per_day:
        raise ValueError("frames_per_day must divide 86400 seconds")
    if not np.isfinite(height_km) or height_km < 0:
        raise ValueError("height_km must be finite and nonnegative")
    grid_path, cache_root = Path(grid_path), Path(cache_root)
    lat, lon, _ = load_grid(grid_path)
    epoch = datetime.datetime(year, 7, 1)
    destination = cache_root/str(year)
    if destination.exists():
        existing = configure_geometry(
            grid_path, cache_root, [year], frames_per_day, use_qd=True
        )["cache_by_year"][str(year)]
        expected = {
            "height_km": height_km,
            "epoch_date": epoch.date().isoformat(),
            "apexpy_version": version,
            "subsolar_height_km": SUBSOLAR_HEIGHT_KM,
        }
        if any(existing.get(key) != value for key, value in expected.items()):
            raise FileExistsError("Existing geometry uses different or incomplete settings. Choose a new --cache-root.")
        print(f"Reused verified cache {destination}")
        return existing

    apex = apexpy.Apex(date=epoch)
    geographic_lat, geographic_lon = np.meshgrid(lat, lon, indexing="ij")
    qlat, qlon = apex.geo2qd(geographic_lat, geographic_lon, height_km)
    qlat = np.asarray(qlat, dtype=np.float64)
    qlon = np.asarray(qlon, dtype=np.float64)
    start = datetime.datetime(year, 1, 1)
    n_frames = (366 if calendar.isleap(year) else 365) * frames_per_day
    step_seconds = 86400 // frames_per_day
    mlt0 = np.empty(n_frames, dtype=np.float64)
    for frame in range(n_frames):
        utc = start + datetime.timedelta(seconds=frame*step_seconds)
        mlt0[frame] = apex.mlon2mlt(0.0, utc, ssheight=SUBSOLAR_HEIGHT_KM)
    if not all(np.isfinite(array).all() for array in (qlat, qlon, mlt0)):
        raise ValueError("ApexPy produced nonfinite geometry values")

    # Compare the affine cache with direct ApexPy calls at sampled longitudes/UTCs.
    locations = np.linspace(0, qlon.size-1, min(9, qlon.size), dtype=int)
    sample_longitudes = qlon.reshape(-1)[locations]
    max_error = 0.0
    for frame in np.linspace(0, n_frames-1, 5, dtype=int):
        utc = start + datetime.timedelta(seconds=int(frame)*step_seconds)
        direct = apex.mlon2mlt(sample_longitudes, utc, ssheight=SUBSOLAR_HEIGHT_KM)
        cached = (sample_longitudes/15.0 + mlt0[frame]) % 24.0
        error = np.abs((cached-direct+12.0) % 24.0-12.0)
        max_error = max(max_error, float(np.max(error)))
    if max_error > 1e-9:
        raise ValueError(f"MLT affine cross-check failed with {max_error:g} hours error")

    coefficient_hashes = {}
    for attribute in ("datafile", "igrf_fn"):
        filename = getattr(apex, attribute, None)
        if filename and Path(filename).is_file():
            coefficient_hashes[Path(filename).name] = digest(filename)
    metadata = {
        "feature_version": FEATURE_VERSION,
        "year": year,
        "grid_sha256": digest(grid_path),
        "shape": [len(lat), len(lon)],
        "frames_per_day": frames_per_day,
        "n_frames": n_frames,
        "start_utc": f"{year}-01-01T00:00:00Z",
        "height_km": float(height_km),
        "epoch_date": epoch.date().isoformat(),
        "epoch_policy": EPOCH_POLICY,
        "apexpy_version": version,
        "method": "ApexPy geo2qd + mlon2mlt",
        "subsolar_height_km": SUBSOLAR_HEIGHT_KM,
        "mlt_formula": "(qd_lon_degrees / 15 + mlon2mlt(0, UTC)) % 24",
        "array_dtype": "float64",
        "mlt_affine_max_error_hours": max_error,
        "coefficient_file_sha256": coefficient_hashes,
        "generator": "prepare_geometry.py public-release reconstruction",
        "generator_sha256": digest(__file__),
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{year}_", dir=cache_root) as directory:
        temporary = Path(directory)
        arrays = {"qd_lat.npy": qlat, "qd_lon.npy": qlon, "mlt0_hours.npy": mlt0}
        for name, array in arrays.items():
            np.save(temporary/name, array)
        metadata["hashes"] = {name: digest(temporary/name) for name in arrays}
        (temporary/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.rename(destination)
    print(f"Generated {destination} with {n_frames} UTC offsets")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-npz", default="data/grid.npz")
    parser.add_argument("--latitude-file", help="One-dimensional NPY in TEC row order")
    parser.add_argument("--longitude-file", help="One-dimensional NPY in TEC column order")
    parser.add_argument("--grid-source", required=True, help="Source product/coordinate fields and preprocessing record")
    parser.add_argument("--cache-root", default="data/geometry")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2012, 2021)))
    parser.add_argument("--height-km", type=float, default=450.0)
    parser.add_argument("--frames-per-day", type=int, default=96)
    args = parser.parse_args()
    grid_provenance = prepare_grid(
        args.grid_npz, args.latitude_file, args.longitude_file, args.grid_source
    )
    for year in args.years:
        prepare_year(args.grid_npz, args.cache_root, year, args.height_km, args.frames_per_day)
    provenance = configure_geometry(
        args.grid_npz, args.cache_root, args.years, args.frames_per_day, use_qd=True
    )
    provenance_path = Path(args.cache_root)/"geometry_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Grid record written to {grid_provenance}")
    print(f"Geometry record written to {provenance_path}")


if __name__ == "__main__":
    main()
