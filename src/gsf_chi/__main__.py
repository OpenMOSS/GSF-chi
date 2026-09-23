"""Run a GSF-chi experiment."""

from __future__ import annotations

import importlib
import sys

TASKS = ("axial_rotation", "axial_ecd", "rs", "ranking", "central_ecd", "moleculenet")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: gsf-chi TASK [--config FILE] [OPTIONS]\n\nTasks: " + ", ".join(TASKS))
        return
    task = sys.argv.pop(1)
    if task not in TASKS:
        raise SystemExit(f"Unknown task {task!r}. Choose from: {', '.join(TASKS)}")
    importlib.import_module("gsf_chi.tasks." + task).main()


if __name__ == "__main__":
    main()
