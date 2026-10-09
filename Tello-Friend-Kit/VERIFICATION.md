# Packaging verification — 8 October 2026

Verified on Windows with Python 3.11.9, using the separate kit directory with spaces in
its path. No physical drone commands or server processes were started for this verification.

- The Windows PowerShell setup created a fresh kit virtual environment with uv and the
  pinned lock. This verification reused cached package downloads; recipients without
  those caches need internet for installation.
- The pip route then bootstrapped pip in that environment, checked the exact dependency
  requirements, built and installed the editable local project and ran the same checks.
  Its runtime packages were already present from the uv installation; a fresh pip download
  of every package was not repeated.
- Both setup routes passed package import checks, **65 offline tests** and Ruff checks.
- `run.cmd doctor`, `run.cmd flight hover` and `run.cmd survey` succeeded when invoked from
  outside the kit directory. The flight/survey commands were offline previews.
- Tests verify that the configured short cage refuses flight before a drone instance is
  created, and check landing cleanup on simulated command failures.
- The SDK PDF matches the SHA-256 recorded in `docs/sdk/SOURCES.md`.
- The packaging script checks ZIP integrity and compares every packed file with its source.
  `MANIFEST.json` records per-file SHA-256 hashes; the archive has a companion `.zip.sha256`.

The validation environment, caches, editable-install metadata and bytecode are removed from
the share folder before delivery. The ZIP uses an explicit file list and includes no flight
photos, logs, videos, firewall backups, installed environment or machine-specific repair scripts.
The receiving PC must run setup to create its own environment.

Actual connectivity, camera reception and flight on your friend's PC/drone remain hardware
checks. The manual macOS/Linux instructions have not been tested on this Windows host.
