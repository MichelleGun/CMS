# Receiving PC setup and troubleshooting

Start with the kit's [README](../README.md). Install dependencies while the PC has internet,
then join the drone's Wi-Fi. SDK commands use `192.168.10.1` by default. The PC should have
a Wi-Fi address such as `192.168.10.2`.

The SDK uses UDP 8889 for commands/responses, 8890 for state and 11111 for video.
If Windows asks about Python network access, allow the interpreter used for this project
on the profile used by the drone network. The virtual-environment launcher may use the
base Python executable, so inspect the actual program path if state/video is blocked.
The setup script does not install firewall rules. An existing explicit block can override
an allow rule; use a narrow allowance for Python, the needed UDP ports and the drone IP
instead of disabling the firewall. This kit does not include the original PC's firewall
repair script or its rule names.

| Symptom | Check |
|---|---|
| Python 3.11 missing | Install Python 3.11 with its launcher, reopen PowerShell and rerun setup |
| Setup download fails | Stay on the internet connection until installation finishes |
| `uv` missing | Setup automatically uses pip; no uv install is required |
| Existing `.venv` has another Python | Extract a fresh copy and rerun setup with Python 3.11 |
| Command timeout | Drone powered on, correct TELLO Wi-Fi, other control apps closed, VPN route checked |
| No state/video | UDP 8890/11111 firewall access; only one SDK/video client running |
| Camera freezes or has distortion | Improve the link; test `observe --seconds 10 --bitrate-mbps 1` on EDU |
| Drone stops responding after a long ground session | Power off to cool, restart, reconnect, then run `check` |
| Low-battery refusal | Recharge; flight starts at 30% or higher |
| Cage clearance refusal | Shipped 43-inch cage is too short for the configured takeoff/roof margin; use ground checks or a suitable larger space with actual measurements |
| Module not found | Use `run.cmd` or this folder's `.venv` interpreter |
| SDK port already used | Close the other drone-control process before running a new one |

For local scripts, direct Tello Wi-Fi needs no internet. To remain online simultaneously,
use Ethernet or phone USB tethering as a second adapter. A phone Wi-Fi hotspot competes
for the PC's Wi-Fi adapter when there is only one adapter.

The basic `check`, `video`, `snapshot`, hover and square tests use standard Tello SDK
commands. Mission pads and the survey/observation modes require Tello EDU. Command support
and firmware differences still need hardware testing on the receiving drone.

One difference is known: the Tello EDU implements SDK 2.0 and answers `setbitrate` with
`unknown command`, because that command is SDK 3.0 (RoboMaster TT). The survey and
observation modes log `video_bitrate_failed` and continue at the drone's default bitrate.
`check` prints the SDK version the drone reports.

If a command fails, its exit status is nonzero. `land` is a normal landing request, not a
motor-cut emergency. Run only one hardware command at a time and keep the aircraft in view.
