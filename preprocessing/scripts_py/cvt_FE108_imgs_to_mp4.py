from pathlib import Path
import subprocess

script_dir = Path(__file__).resolve().parents[1] / "scripts"
utils_dir = Path(__file__).resolve().parents[1] / "utils"
script_path = script_dir / "cvt_FE108_imgs_to_mp4.sh"


subprocess.run(["bash", str(script_path)], cwd=str(utils_dir), check=True)
