# warehouse-forklift


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
