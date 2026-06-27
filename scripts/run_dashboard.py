"""Launch the Streamlit dashboard.

Why a launcher instead of `streamlit run ...` directly? On Windows, importing
torch *after* pyarrow (a Streamlit dependency) fails with
`OSError: [WinError 1114] ... c10.dll`: pyarrow's bundled DLLs get into the
process first and torch's C extensions then fail to initialise. `streamlit run`
imports Streamlit (hence pyarrow) before it executes the app script, so the
order can't be fixed from inside dashboard.py.

This launcher imports torch first — loading its DLLs before pyarrow's — then
hands off to the Streamlit CLI in the same process. Order preserved, conflict
avoided.

    python scripts/run_dashboard.py
"""
from __future__ import annotations

import torch  # noqa: F401  -- must load torch's DLLs before streamlit/pyarrow

import sys
from pathlib import Path

from streamlit.web import cli as stcli

if __name__ == "__main__":
    dashboard = Path(__file__).resolve().parents[1] / "src" / "pointcloud_clf" / "dashboard.py"
    # fileWatcherType=none stops Streamlit's watcher from enumerating torch's
    # C-extension paths (the noisy "torch.classes" warning). Extra CLI args are
    # passed through, e.g. `python scripts/run_dashboard.py --server.port 8600`.
    sys.argv = ["streamlit", "run", str(dashboard), "--server.fileWatcherType=none", *sys.argv[1:]]
    sys.exit(stcli.main())
