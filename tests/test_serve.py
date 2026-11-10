"""Unit tests for the service job script (deploy/tacc/service/qwen-service/serve.py) that need no GPU."""
import importlib
import importlib.util
import sys
from pathlib import Path

SERVE = Path(__file__).resolve().parents[1] / "deploy/tacc/service/qwen-service/serve.py"


def load_serve():
    # serve.py imports the staged copy of the client as `client`; point that name at the real module.
    sys.modules.setdefault("client", importlib.import_module("inference.client"))
    spec = importlib.util.spec_from_file_location("serve_under_test", SERVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_gpu_csv_reads_numbers_and_tolerates_missing_power():
    serve = load_serve()
    gpus = serve.parse_gpu_csv("0, 97, 20480, 24576, 312.4\n1, 95, 20100, 24576, [N/A]\n")
    assert gpus == [
        {"index": 0, "util_pct": 97.0, "mem_used_mib": 20480.0, "mem_total_mib": 24576.0, "power_w": 312.4},
        {"index": 1, "util_pct": 95.0, "mem_used_mib": 20100.0, "mem_total_mib": 24576.0, "power_w": None},
    ]


def test_parse_gpu_csv_ignores_noise():
    assert load_serve().parse_gpu_csv("NVIDIA-SMI has failed\n\n") == []
