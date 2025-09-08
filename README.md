# Agriculture

A lightweight workspace for agriculture-related analysis, experiments, and notebooks.

## Overview

Use this repo to keep Jupyter notebooks, small scripts, and related assets together while you iterate in VS Code.

## Quick Start

1. Open the folder in VS Code.
2. (Optional) Create a Python virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate` (macOS/Linux) or `.venv\\Scripts\\Activate` (Windows)
3. Install dependencies if you have a `requirements.txt`:
   - `pip install -r requirements.txt`
4. Run notebooks with the VS Code Jupyter extension (select the `.venv` interpreter if prompted).

## Repo Layout

- Notebooks: `*.ipynb`
- Data (optional): `data/`
- Scripts (optional): `scripts/`

## Notes

- Keep large/raw data out of Git when possible. Prefer a `data/` folder in `.gitignore` or use cloud storage.
- Add lightweight environment and run instructions here as the project grows.

