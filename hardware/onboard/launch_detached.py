"""
Start a long job so nothing else can stop it by accident: its own hidden console, in a new process
group (Windows ignores Ctrl+C there), output to files. The launcher exits at once.

    python hardware/onboard/launch_detached.py results/hoop/loop_laps5 -- hardware/onboard/loop_night.py --tag laps5 ...

The first argument is the output prefix (<prefix>.stdout.txt / .stderr.txt); everything after `--`
is passed to this Python. Started this way because every run of round 5 stopped at once on a Ctrl+C
that reached the console they shared (2026-10-14 12:23).
"""
import subprocess
import sys

if __name__ == "__main__":
    if "--" not in sys.argv[2:]:
        sys.exit(__doc__)
    prefix = sys.argv[1]
    args = sys.argv[sys.argv.index("--") + 1:]
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
             | subprocess.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0)
    p = subprocess.Popen([sys.executable, *args], stdout=open(prefix + ".stdout.txt", "w"),
                         stderr=open(prefix + ".stderr.txt", "w"), creationflags=flags,
                         start_new_session=sys.platform != "win32")
    print(p.pid)
