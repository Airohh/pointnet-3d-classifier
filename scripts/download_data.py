"""Download + extract ModelNet10 into data/raw/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointcloud_clf.data import download_modelnet10  # noqa: E402

if __name__ == "__main__":
    root = download_modelnet10()
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"ModelNet10 ready at {root}")
    print(f"classes ({len(classes)}): {classes}")
