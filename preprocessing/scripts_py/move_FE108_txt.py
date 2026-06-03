from pathlib import Path
import subprocess

script_dir = Path(__file__).resolve().parents[1] / "scripts"
utils_dir = Path(__file__).resolve().parents[1] / "utils"
script_path = script_dir / "move_txt.sh"


subprocess.run(["bash", str(script_path)], cwd=str(utils_dir), check=True)
