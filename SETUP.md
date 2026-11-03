# Setup & Workflow Guide

## Repository
- **GitHub**: https://github.com/Purdue-Huang-Lab/Automation (public)
- **Main branch**: `main`
- **Python version**: 3.10

---

## 1. Push an update to a feature branch (safe testing)

Use this when you want to save work-in-progress without touching `main`.

```powershell
# Create and switch to a new branch (first time)
git checkout -b your-branch-name

# Or switch to an existing branch
git checkout your-branch-name

# Stage, commit, and push
git add .
git commit -m "describe what changed"
git push -u origin your-branch-name   # first push
git push                               # subsequent pushes
```

To merge into `main` later, open a Pull Request on GitHub, or:

```powershell
git checkout main
git merge your-branch-name
git push
```

---

## 2. Push an update to main

Use this for confirmed, working changes.

```powershell
git checkout main
git add .
git commit -m "describe what changed"
git push
```

**What gets excluded automatically** (via `.gitignore`):
- `.venv/`, `__.venv/` — virtual environments
- `__pycache__/`, `*.pyc` — Python cache
- `*.csv`, `*.sif`, `*.h5`, `*.npy`, `*.npz`, `*.mat`, `*.dat` — measurement data
- `code by weijian/`, `0402plgatetest/` — excluded folders
- `.claude/` — Claude Code local settings
- `___All_Errors.txt`, `.env`, `.DS_Store`

---

## 3. Configure a new environment

### Step 1 — Clone the repo

```powershell
git clone https://github.com/Purdue-Huang-Lab/Automation.git
cd Automation
```

The repo is public, so cloning needs no login. Pushing requires write access to the repo in the
Purdue-Huang-Lab organization; on the first `git push`, Git Credential Manager (bundled with Git for
Windows) opens a browser to sign in to GitHub. Never put a token in the remote URL.

### Step 2 — Create and activate the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Make sure `python` points to **Python 3.10**. Check with `python --version`.

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Install spinnaker-python manually (FLIR camera SDK)

`spinnaker-python` is **not** in `requirements.txt` because it must be installed from a local `.whl` file provided by the FLIR SDK installer. After installing the FLIR Spinnaker SDK:

```powershell
pip install "C:\Program Files\Python310\PySpin\spinnaker_python-4.3.0.189-cp310-cp310-win_amd64.whl"
```

The exact path may differ — check your FLIR SDK installation directory.

### Step 5 — Verify

Run any of the launcher scripts to confirm everything works:

```powershell
.\run_keithley_gui.ps1
```

---

## Notes for coding agents

- **Never commit** `*.csv`, `*.sif`, or other data files — they are measurement outputs.
- **Never commit** `.venv/` or `__.venv/` — recreate from `requirements.txt`.
- **Never commit** `.claude/settings.local.json` — it is machine-specific.
- `ph300/740 irf 20260311.csv` is a calibration file (not data) and **is** tracked — do not add `ph300/*.csv` to `.gitignore`.
- If `requirements.txt` needs updating, run `pip freeze > requirements.txt` with the venv activated, then fix encoding if on Windows PowerShell:
  ```powershell
  [System.IO.File]::WriteAllText("$PWD\requirements.txt", (pip freeze | Out-String), [System.Text.UTF8Encoding]::new($false))
  ```
- Branch `main` is the production branch. Use a feature branch for experimental changes.
