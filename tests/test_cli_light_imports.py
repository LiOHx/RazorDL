import subprocess
import sys


def test_cli_import_does_not_pull_torch():
    """`razordl --help` must stay light: importing the CLI must not import torch."""
    code = (
        "import sys\n"
        "import razordl.cli\n"
        "import razordl.ops.snapshot\n"
        "heavy = sorted(m for m in ('torch', 'peft', 'ray', 'tensordict', 'vllm') if m in sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
