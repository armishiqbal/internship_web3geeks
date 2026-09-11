"""Executes day5.ipynb cells sequentially and writes back clean outputs into day5.ipynb."""

import io
import sys
import contextlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK_PATH = HERE / "day5.ipynb"

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Loaded {NOTEBOOK_PATH} with {len(nb['cells'])} cells.")

shared_globals = {
    "__name__": "__main__",
    "__file__": str(NOTEBOOK_PATH),
}

exec_count = 1

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        print(f"\n[*] Executing Cell {idx} (Task Code)...")

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(source, shared_globals)
        except Exception as e:
            print(f"    [!] Error in Cell {idx}: {e}")
            raise e

        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()

        # Format stdout text into lines for Jupyter
        lines = stdout_text.splitlines(keepends=True)
        cell["outputs"] = [
            {
                "output_type": "stream",
                "name": "stdout",
                "text": lines,
            }
        ]
        cell["execution_count"] = exec_count
        exec_count += 1
        print(f"    [OK] Output captured: {len(lines)} lines ({len(stdout_text)} chars). Stderr: {len(stderr_text)} chars.")

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nSuccessfully executed all cells and saved {NOTEBOOK_PATH} with zero errors!")
