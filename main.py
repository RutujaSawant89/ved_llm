"""TrustLM: A Custom Large Language Model for Reliable AI Responses.

Environment & Project Foundation Verification Runner.
Executes systematic health checks to confirm:
  1. Python runtime environment
  2. PyTorch installation & compute device discovery
  3. Core research dependencies (tiktoken, numpy, pandas, matplotlib, seaborn, tqdm)
  4. Project directory structure integrity
  5. Configuration system initialization and validation
  6. Deterministic seeding and utility operations
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Terminal styling helpers
class Style:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(title: str) -> None:
    line = "=" * 70
    print(f"\n{Style.CYAN}{Style.BOLD}{line}")
    print(f" {title.center(68)}")
    print(f"{line}{Style.RESET}")


def print_status(component: str, status: str, details: str = "", level: str = "PASS") -> None:
    color = Style.GREEN if level == "PASS" else (Style.YELLOW if level == "WARN" else Style.RED)
    tag = f"[{level}]"
    print(f"  {color}{tag:<6}{Style.RESET} {Style.BOLD}{component:<28}{Style.RESET}: {details}")


def check_python_environment() -> Tuple[bool, str]:
    """Verify Python version satisfies minimum requirement (>= 3.10)."""
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro} ({platform.python_implementation()})"
    if (major, minor) >= (3, 10):
        return True, f"Python {version_str} - OK"
    return False, f"Python {version_str} - Requires Python >= 3.10"


def check_pytorch_environment() -> Tuple[bool, str, Dict[str, str]]:
    """Verify PyTorch installation, device acceleration, and basic tensor ops."""
    try:
        import torch

        version = torch.__version__
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            device_info = f"CUDA Available: {device_name} ({device_count} device(s))"
            dev = torch.device("cuda")
        else:
            device_info = "CPU only (CUDA not detected - suitable for development & testing)"
            dev = torch.device("cpu")

        # Sanity check tensor operations on selected device
        a = torch.randn(32, 64, device=dev)
        b = torch.randn(64, 32, device=dev)
        c = torch.matmul(a, b)
        assert c.shape == (32, 32)

        details = {
            "version": version,
            "device": str(dev),
            "info": device_info,
        }
        return True, f"PyTorch {version} on {dev} (matmul sanity verified)", details
    except Exception as exc:
        return False, f"PyTorch verification failed: {exc}", {}


def check_required_dependencies() -> List[Tuple[str, bool, str]]:
    """Check availability and versions of all required third-party packages."""
    required_packages = [
        ("torch", "PyTorch deep learning framework"),
        ("numpy", "Numerical computing library"),
        ("tiktoken", "Fast byte-pair encoding tokenizer"),
        ("tqdm", "Progress bar utility for training loops"),
        ("pandas", "Data manipulation and analysis"),
        ("matplotlib", "Plotting and visualization"),
        ("seaborn", "Statistical data visualization"),
    ]

    results = []
    for pkg_name, description in required_packages:
        try:
            mod = importlib.import_module(pkg_name)
            version = getattr(mod, "__version__", "unknown")
            results.append((pkg_name, True, f"v{version} ({description})"))
        except ImportError as exc:
            results.append((pkg_name, False, f"MISSING - {exc}"))
    return results


def check_project_structure(project_root: Path) -> List[Tuple[str, bool, str]]:
    """Verify that all required modular directories and foundational files exist."""
    required_dirs = [
        "data",
        "tokenizer",
        "model",
        "training",
        "generation",
        "verification",
        "trust",
        "memory",
        "unlearning",
        "evaluation",
        "utils",
        "tests",
    ]

    required_files = [
        "config.py",
        "requirements.txt",
        "README.md",
        "main.py",
    ]

    results = []
    for d in required_dirs:
        dir_path = project_root / d
        exists = dir_path.is_dir()
        has_init = (dir_path / "__init__.py").is_file()
        status = "Directory exists"
        if exists and has_init:
            status += " with __init__.py"
        results.append((f"dir: {d}/", exists, status))

    for f in required_files:
        file_path = project_root / f
        exists = file_path.is_file()
        size_kb = f"{file_path.stat().st_size / 1024:.1f} KB" if exists else "0 KB"
        results.append((f"file: {f}", exists, f"File exists ({size_kb})"))

    return results


def check_configuration_system(project_root: Path) -> Tuple[bool, str]:
    """Import and validate the central configuration system."""
    try:
        # Ensure project root is in sys.path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from config import TrustLMConfig, get_default_config

        cfg = get_default_config()
        cfg.validate()

        # Test dictionary serialization
        data = cfg.to_dict()
        assert "model" in data, "Missing 'model' section in config serialization"
        assert "verification" in data, "Missing 'verification' section in config serialization"
        assert "trust" in data, "Missing 'trust' section in config serialization"
        assert "unlearning" in data, "Missing 'unlearning' section in config serialization"
        assert "memory" in data, "Missing 'memory' section in config serialization"

        summary = (
            f"Validated [d_model={cfg.model.d_model}, layers={cfg.model.n_layers}, "
            f"heads={cfg.model.n_heads}, ctx={cfg.model.context_length}, "
            f"batch_size={cfg.training.batch_size}]"
        )
        return True, summary
    except Exception as exc:
        return False, f"Configuration check failed: {exc}"


def check_utilities(project_root: Path) -> Tuple[bool, str]:
    """Test utility functions such as seed setting and device retrieval."""
    try:
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from utils import get_device, set_seed

        set_seed(42)
        dev = get_device()
        return True, f"Deterministic seed set (42); default device resolved to {dev}"
    except Exception as exc:
        return False, f"Utilities check failed: {exc}"


def main() -> int:
    """Run all environment and project foundation checks."""
    print_header("TrustLM: Project Foundation & Environment Verification")
    print(
        f"{Style.BOLD}Research Title:{Style.RESET} TrustLM: A Custom Large Language Model for Reliable AI Responses\n"
        f"{Style.BOLD}Core Principle:{Style.RESET} Pure PyTorch from scratch - No external LLM wrappers\n"
        f"{Style.BOLD}Research Goals:{Style.RESET} 1. Response Verification | 2. Trust Scoring | 3. Selective Unlearning\n"
        f"{Style.BOLD}Component:{Style.RESET}      4. Controlled Modular Memory"
    )

    project_root = Path(__file__).resolve().parent
    failures: int = 0
    warnings: int = 0

    # 1. Python Environment Check
    print_header("1. Python Runtime Environment")
    py_ok, py_msg = check_python_environment()
    print_status("Python Version", "PASS" if py_ok else "FAIL", py_msg, level="PASS" if py_ok else "FAIL")
    if not py_ok:
        failures += 1

    # 2. PyTorch & Device Discovery
    print_header("2. PyTorch & Compute Acceleration")
    torch_ok, torch_msg, torch_info = check_pytorch_environment()
    print_status("PyTorch Core", "PASS" if torch_ok else "FAIL", torch_msg, level="PASS" if torch_ok else "FAIL")
    if torch_ok:
        print_status("Compute Device", "INFO", torch_info.get("info", ""), level="PASS")
    else:
        failures += 1

    # 3. Third-party Dependencies
    print_header("3. Core Research Dependencies")
    dep_results = check_required_dependencies()
    for pkg_name, dep_ok, dep_msg in dep_results:
        lvl = "PASS" if dep_ok else "FAIL"
        print_status(f"Package '{pkg_name}'", lvl, dep_msg, level=lvl)
        if not dep_ok:
            failures += 1

    # 4. Project Directory Structure
    print_header("4. Project Structure Integrity")
    struct_results = check_project_structure(project_root)
    for name, s_ok, s_msg in struct_results:
        lvl = "PASS" if s_ok else "FAIL"
        print_status(name, lvl, s_msg, level=lvl)
        if not s_ok:
            failures += 1

    # 5. Configuration System
    print_header("5. Configuration System Sanity")
    cfg_ok, cfg_msg = check_configuration_system(project_root)
    print_status("Config System", "PASS" if cfg_ok else "FAIL", cfg_msg, level="PASS" if cfg_ok else "FAIL")
    if not cfg_ok:
        failures += 1

    # 6. Utilities & Seeding
    print_header("6. Utilities & Determinism")
    utils_ok, utils_msg = check_utilities(project_root)
    print_status("Seed & Device", "PASS" if utils_ok else "FAIL", utils_msg, level="PASS" if utils_ok else "FAIL")
    if not utils_ok:
        failures += 1

    # Summary Report
    print_header("Environment Verification Summary")
    if failures == 0:
        print(f"\n{Style.GREEN}{Style.BOLD}>>> ALL FOUNDATION CHECKS PASSED SUCCESSFULLY! <<<{Style.RESET}")
        print("Project is ready for Phase 2 (Tokenizer & Custom Dataset Pipeline).")
        return 0
    else:
        print(f"\n{Style.RED}{Style.BOLD}>>> FOUNDATION VERIFICATION COMPLETED WITH {failures} FAILURE(S) <<<{Style.RESET}")
        print("Please resolve the failed checks before proceeding to subsequent phases.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
