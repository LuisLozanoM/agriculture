# Agriculture

A lightweight workspace for agriculture-related analysis, experiments, and notebooks.

## Overview

Use this repo to keep Jupyter notebooks, small scripts, and related assets together while you iterate in VS Code.

## Prerequisites

- **Python 3.10+** – confirm with `python3 --version`. On macOS install via Homebrew (`brew install python@3.12`).
- **uv** – install with `curl -Ls https://astral.sh/uv/install.sh | sh` (macOS/Linux) or follow https://docs.astral.sh/uv/getting-started/ on Windows.
- **Ollama** – install from https://ollama.com/download or run `curl -fsSL https://ollama.com/install.sh | sh` (macOS/Linux) and start with `ollama serve`.
- **Google Earth Engine access** – run `earthengine authenticate` (browser flow) or rely on `ee.Authenticate()` during app startup.
- **Optional secrets** – place credentials in `.env`; Streamlit loads it automatically.

## Quick Start

1. Open the folder in VS Code.
2. Install the environment with `uv sync` (installs deps into `.venv`).
3. Launch Streamlit with `uv run streamlit run st_chat1.py`.
4. For notebooks, pick the `Python (agriculture)` kernel that `uv sync` registered.

## Agent/Handoff Notes

- Rehydrate the project on a new machine by completing **Prerequisites**, running `uv sync`, then `uv run streamlit run st_chat1.py`.
- Keep Ollama running locally on port 11434 for `st_chat1.py`: start with `ollama serve` and verify via `curl http://localhost:11434/api/tags`.
- Earth Engine requests expect project `ee-lalozanom`; adjust in the Streamlit sidebar if you need another project.
- If the notebook kernel is missing, rerun `.venv/bin/python -m ipykernel install --user --name agriculture --display-name "Python (agriculture)"`.

## Repo Layout

- Notebooks: `*.ipynb`
- Data (optional): `data/`
- Scripts (optional): `scripts/`

## Notes

- Keep large/raw data out of Git when possible. Prefer a `data/` folder in `.gitignore` or use cloud storage.
- Add lightweight environment and run instructions here as the project grows.
