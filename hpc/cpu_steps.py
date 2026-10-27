"""Several steps in one CPU job (cpu.sbatch has one command): python hpc/cpu_steps.py <steps file in the repo>.
Lines `python <args>` run with this interpreter; a leading `-` reports the exit code without failing the job."""
import subprocess
import sys
import time

for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    soft = line.startswith("-")
    cmd = line.lstrip("-").replace("python", sys.executable, 1)
    print(f"=== step start {time.time():.1f} {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    print(f"=== step done {time.time():.1f} rc {rc}", flush=True)
    if rc and not soft:
        sys.exit(rc)
