# warehouse-forklift

## Sample outputs

### R2 — Forklift slowdown detection

![R2 forklift slowdown detection output](pics/r2.png)

### R6 — Phone usage detection

![R6 phone usage detection output](pics/r6.png)

### R7 — Unsafe double-action detection

![R7 unsafe double-action detection output](pics/r7.png)

## Setup and usage

```powershell
uv sync

# Run tools without manually activating the environment
uv run --frozen python r7/test-tracking/scripts/r7_demo.py
uv run --frozen python r2/test-tracking/scripts/r2_demo.py
uv run --frozen python r2/test-tracking/scripts/track_forklift.py
uv run --frozen python r6-2/infer_camera.py

# Optional activation for an IDE terminal
.\.venv\Scripts\Activate.ps1
```

Select `.venv\Scripts\python.exe` as the Python/Jupyter interpreter. The lock
uses Python 3.12, Ultralytics 8.4.116, and the PyTorch CUDA 13.0 wheels shared by
the detection, pose, tracking, segmentation, notebook, and ONNX workflows.
