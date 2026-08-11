# warehouse-forklift

The repository uses one shared `uv` environment at `.venv/` for `r2`, `r6-2`,
and `r7`.

```powershell
# Create/update the shared environment from the repository root
uv sync

# Run tools without manually activating the environment
uv run python r7/test-tracking/scripts/r7_demo.py --help
uv run pytest r6-2/tests

# Optional activation for an IDE terminal
.\.venv\Scripts\Activate.ps1
```

Select `.venv\Scripts\python.exe` as the Python/Jupyter interpreter. The lock
uses Python 3.12, Ultralytics 8.4.116, and the PyTorch CUDA 13.0 wheels shared by
the detection, pose, tracking, segmentation, notebook, and ONNX workflows.
