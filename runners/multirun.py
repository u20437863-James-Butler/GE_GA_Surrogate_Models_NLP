import subprocess
import sys

# Ensure user provided at least one argument
if len(sys.argv) < 2:
    print("Usage: python controller.py param1 param2 param3 ...")
    sys.exit(1)

# Get parameters passed to controller.py (skip sys.argv[0], which is script name)
params = sys.argv[1:]

for param in params:
    # This constructs the command to activate conda env and run runner.py with a param
    cmd = f"python .\\runners\\runner.py --config {param}"
    
    # Call the command
    result = subprocess.run(cmd, shell=True)

    # Optional: check return code
    if result.returncode != 0:
        print(f"Run with param '{param}' failed with return code {result.returncode}")
