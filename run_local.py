#!/usr/bin/env python3
"""
Run RAG backend (serves API + UI on one server).
From project root:  python run_local.py
Then open:  http://localhost:5000
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(ROOT, "code")
REQUIREMENTS = os.path.join(CODE_DIR, "requirements.txt")


def main():
    os.chdir(ROOT)
    try:
        import flask  # noqa: F401
    except ModuleNotFoundError:
        print("Installing dependencies (run once): pip install -r code/requirements.txt")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS], cwd=ROOT)
        if r.returncode != 0:
            print("Install failed. Run manually: pip install -r code/requirements.txt")
            return 1
        print()
    print("Open http://localhost:5000 in your browser.")
    print("Stop with Ctrl+C.\n")
    r = subprocess.run([sys.executable, os.path.join(CODE_DIR, "app.py")], cwd=ROOT, env={**os.environ, "PORT": "5000"})
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
