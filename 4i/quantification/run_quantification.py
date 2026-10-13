"""Run quantification directly from this checkout, without installing the package."""
from pathlib import Path
import sys

# Resolve local sources relative to this file, including in Slurm workers.
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir / "src"))
vistools_dir = script_dir.parents[2] / "MYONIS-VisTools"
if (vistools_dir / "myonis_vis" / "__init__.py").is_file():
    sys.path.insert(0, str(vistools_dir))

from fouri_quantification.cli import main


if __name__ == "__main__":
    arguments = sys.argv[1:]
    commands = {"run", "prepare", "submit-batch", "worker", "worker-batch", "finalize", "validate", "report"}
    raise SystemExit(main(arguments if arguments and arguments[0] in commands else ["run", *arguments]))
