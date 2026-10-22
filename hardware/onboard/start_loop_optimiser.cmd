@echo off
rem Start (or resume) the loop-policy optimiser, detached and at below-normal priority.
rem Called at logon by "CMS loop optimiser.cmd" in your Startup folder (Win+R, shell:startup);
rem also fine to double-click. loop_forever.py refuses to start a second copy, so running this
rem twice is harmless.
rem Stop after the current round:   type nul > results\hoop\mg_loop.stop
rem Stop starting it at logon:      delete "CMS loop optimiser.cmd" from shell:startup
cd /d "%~dp0..\.."
set PY=%USERPROFILE%\miniconda3\envs\drones\python.exe
if not exist "%PY%" set PY=python
"%PY%" hardware\onboard\launch_detached.py results\hoop\mg_loop -- hardware\onboard\loop_forever.py
