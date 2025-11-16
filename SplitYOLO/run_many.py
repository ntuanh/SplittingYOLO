import subprocess, time, sys

venv_python = sys.executable
log_file = "measure_log.txt"

for i in range(50):
    print(f"[{i+1}/50] Running main.py in venv...")
    result = subprocess.run(
        [venv_python, "main.py"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(result.stdout + ",")
