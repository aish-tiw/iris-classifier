# tests/test_train.py
# Basic pytest example for the Iris classifier project

import subprocess
import sys
from pathlib import Path


def test_train_runs():
    
    outputs_dir = Path("outputs")

    
    if outputs_dir.exists():
        for f in outputs_dir.glob("*"):
            f.unlink()

    
    result = subprocess.run(
        [sys.executable, "src/train.py"],
        capture_output=True,
        text=True
    )

    
    assert result.returncode == 0, f"train.py failed: {result.stderr}"

    
    assert (outputs_dir / "confusion_matrix.png").exists()
    assert (outputs_dir / "model.joblib").exists()

