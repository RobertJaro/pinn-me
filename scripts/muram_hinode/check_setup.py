"""Check the stored Hinode forward contract and gradients without training."""
from pathlib import Path
from prom3theus.muram import config as _schema_registration  # noqa: F401
from prom3theus.muram import observation as _adapter_registration  # noqa: F401
from prom3theus.config import load_config
from prom3theus.application.joint_runner import run_joint_dry_run

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    result = run_joint_dry_run(load_config(root / "configs/muram/hinode/mhs.yaml"))
    print(result.report_path)
