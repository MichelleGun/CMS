# Tello friend test kit

Self-contained source project for one Tello/Tello EDU, with Python setup, camera and
telemetry checks, supervised flight scripts, offline tests, and the official EDU SDK guide.
The Windows instructions use Python 3.11 (64-bit). Setup requires internet; after setup,
the installed scripts communicate with the drone locally over Wi-Fi.

## Set up on Windows

1. Extract the entire ZIP to a writable folder, such as `Documents\Tello-Friend-Kit`.
   Run scripts from the extracted folder, not from inside the ZIP.
2. Install Python 3.11 if needed. The original environment was verified with
   [Python 3.11.9, Windows 64-bit installer](https://www.python.org/downloads/release/python-3119/).
   Include the Python launcher during installation. A newer installed 3.11 patch is also
   accepted; this kit's dependency lock requires Python 3.11.
3. While connected to the internet, open PowerShell in the extracted folder and run:

   ```powershell
   .\setup.cmd
   ```

Setup creates `.venv` in this folder, installs pinned packages, checks imports, runs
offline tests and checks the Python source. It uses `uv` if already installed and otherwise
uses Python's pip. You do not need to install uv. To explicitly use pip:

```powershell
.\setup.cmd -UsePip
```

The wrapper permits only this setup process to run its PowerShell script; it does not
change the machine's execution-policy setting. Setup makes no drone connection, flight,
firewall, or Wi-Fi changes. Source files can be edited after installation.

## Connect and check the drone

1. Power on the drone and join its `TELLO-xxxxxx` Wi-Fi from the PC. Close the phone Tello
   app and other programs controlling the aircraft. Only one SDK program should run at a time.
2. Run the connection check:

   ```powershell
   .\run.cmd check
   ```

3. Try the camera and stationary telemetry:

   ```powershell
   .\run.cmd video --seconds 20
   .\run.cmd snapshot
   .\run.cmd telemetry --seconds 10
   ```

These commands do not take off. The camera preview closes with Q/Esc. Logs, photos and
recordings are saved under `data/` in this folder. Use USB phone tethering or Ethernet for
internet while the PC's Wi-Fi is connected to Tello. Internet is not needed for installed
drone scripts; a continuing online chat requires a separate internet connection.

## Flight and the included cage settings

The shipped `config/tello.toml` retains the known test cage dimensions: **43 inches per
side = 109.22 cm**, with a 30 cm project roof-clearance margin. That leaves 79.22 cm of
nominal height. Automatic takeoff was observed around 80 cm and cannot be set to a lower
height by these scripts. **Real-flight commands currently refuse takeoff in this cage.**
Ground connection, camera and telemetry checks still work.

If you test in a different enclosure, edit `[cage]` to its actual measured dimensions.
The clearance is a project margin, not a manufacturer specification. The stored width
and length do not implement horizontal obstacle avoidance. Check the physical space for
the requested path; a camera candidate or SDK acknowledgement does not establish clearance.

Preview a hover plan offline:

```powershell
.\run.cmd flight hover
```

In an appropriate clear test area whose measurements satisfy the configuration, fit guards,
keep the drone in sight and explicitly enable the short test:

```powershell
.\run.cmd flight hover --fly
```

This counts down for five seconds, takes off, hovers for five seconds and requests landing.
Ctrl+C cancels before takeoff or requests landing if takeoff was attempted. A Wi-Fi loss,
process termination or failed landing command can prevent recovery. Start battery must
be at least 30%; the test requests landing below 20%. After reviewing a stable hover,
`flight square` previews a 50 cm square; `flight square --fly` executes it at 20 cm/s.
The commanded square is not an independently measured trajectory.

## Script reference

| Command after `run.cmd` | Action |
|---|---|
| `doctor` | Check Python and package imports offline |
| `check` | Connect, read battery/state and query SDK version |
| `video --seconds 20` | Ground camera preview |
| `video --record --seconds 20` | Ground preview with an MP4 recording |
| `snapshot` | Save a camera JPEG and state JSON |
| `telemetry --seconds 10` | Save ground state CSV |
| `telemetry --seconds 10 --mission-pads` | Ground mission-pad detection; EDU required |
| `observe --seconds 10 --bitrate-mbps 1` | Live blue-ring candidates and pad telemetry; EDU required |
| `flight hover` / `flight square` | Preview plans offline |
| `flight hover --fly` / `flight square --fly` | Explicit real flight, subject to cage checks |
| `survey` | Preview a camera survey offline |
| `survey --fly` | EDU camera survey with yaw turns and landing |
| `survey --search-forward --landing-pad 3 --fly` | EDU raised forward/side survey and detected-pad return |
| `land` | Request normal landing; no takeoff |

`observe` finds possible blue outlines in image coordinates. It does not autonomously
fly through a circle or establish its physical size. `survey --search-forward` adds a
20 cm rise, so it needs additional roof clearance. Survey/observation refuse a starting
mainboard temperature above the project's 80°C limit. The survey requests landing above
85°C or on missing/stale telemetry, invalid range or reported height outside 30–120 cm.
Blocking SDK calls can delay an abort until they finish. Pad numbers belong to each setup;
pad 3 is not assumed to exist on another test site.

To choose a different configuration, place `--config` before the command:

```powershell
.\run.cmd --config .\config\tello.toml check
```

## Environment and developer commands

The runtime pins are DJITelloPy 2.5.0, OpenCV 4.14.0.94, NumPy 2.4.6, PyAV 18.1.0 and
Pillow 12.3.0. `requirements-dev.txt` also pins pytest, Ruff and their dependencies.
`uv.lock` records distribution hashes for the optional uv setup route. Installed packages
are recreated on the receiving PC; a virtual environment is not part of the ZIP.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m tello_lab --help
```

All source is in `src/tello_lab/`. Offline tests use fake drones and do not transmit
flight commands. The kit includes no running server, automatic takeoff on import, RL policy
or simulation. Start commands explicitly and stop the active command with Ctrl+C.

For macOS/Linux with Python 3.11 installed, use the equivalent manual setup:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m tello_lab doctor
.venv/bin/python -m pytest -q
.venv/bin/python -m tello_lab check
```

The `.cmd`/PowerShell wrappers are for Windows; macOS/Linux use the Python module directly.
Those platforms have not been verified on this Windows host.

## Troubleshooting and references

See [docs/setup.md](docs/setup.md) for Wi-Fi, firewall and camera troubleshooting.
The unchanged [official SDK 2.0 PDF](docs/sdk/Tello_SDK_2.0_User_Guide.pdf) is included;
[its provenance](docs/sdk/SOURCES.md) records the official download and checksum.
DJITelloPy documentation is at [its API reference](https://djitellopy.readthedocs.io/en/latest/).
Third-party packages and the PDF retain their respective licences and copyright.

`MANIFEST.json` lists the shared files and their SHA-256 hashes. Read `VERIFICATION.md`
for the checks performed before packaging.
