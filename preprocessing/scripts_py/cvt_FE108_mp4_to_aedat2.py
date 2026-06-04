import subprocess
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm
from preprocessing.settings.local import EnvironmentSettings

script_dir = Path(__file__).resolve().parents[1] / "scripts"
script_path = script_dir / "cvt_FE108_mp4_to_aedat2.sh"

def processing(mp4_dir):  # 传入数据集目录
    log_dir = Path(__file__).resolve().parents[1] / "logs"  # log目录
    log_dir.mkdir(parents=True, exist_ok=True)

    # 如果log_path已经存在, 就按顺序新建
    def next_log_path(prefix="v2e", suffix=".log"):
        i = 1
        while True:
            path = log_dir / f"{prefix}{i}{suffix}"
            if not path.exists():
                return path
            i += 1
    log_path = next_log_path()

    all_roots = []
    for root, dirs, files in os.walk(mp4_dir):  # 递归遍历数据集
        if glob(os.path.join(root, "raw.mp4")):  # 如果当前目录下有raw.mp4, 保存当前路径
            all_roots.append(root)

    with open(log_path, "a") as f:
        for idx, root in enumerate(tqdm(all_roots, desc="Converting mp4 to aedat2"), 1):
            # 转换raw.mp4
            subprocess.run(["bash", str(script_path)], cwd=str(root), check=True, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    EnvSettings = EnvironmentSettings()
    mp4_dir = EnvSettings.v2e_data_path
    processing(mp4_dir)
