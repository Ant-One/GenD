# Agent Instructions & Gotchas — GenD

This document contains high-signal, repo-specific context, setup information, and developer workflows to help agents ramp up immediately and avoid common mistakes.

---

## ⚠️ Critical Operational Gotchas (Must Read)

### 1. Interactive Hanging on Existing Runs
In `src/utils/checks.py`, if a run directory already exists and the name does not contain `"tmp"`, the environment **will prompt for interactive input** (`key = input()`), asking to `Enter R to replace`. This will cause any non-interactive agent execution to **hang indefinitely**.
* **Prevention:** Always do one of the following:
  - Add `"tmp"` to your run/experiment name (e.g., `test-tmp` or `--run_name tmp-run`).
  - Pass `--remove_if_run_exists=True` or `--throw_exception_if_run_exists=True` when executing commands that support config parameters.

### 2. Dataset Path and Binary Labelling
`src/dataset/dataset.py` extracts binary labels directly from the image path string:
* The path must strictly contain either `"real"` or `"fake"` as a substring.
* If a path does not contain either of these, the script aborts immediately with `sys.exit("I do not know how to stop the program otherwise.")`.
* Standard path structure expected: `... / <dataset_name> / <fake/real> / <source_name> / <video_name> / <frame_name>`.

---

## 🛠️ Developer Commands

### Environment Setup
The repository leverages Conda and `uv` for python dependencies:
```bash
conda create --name GenD python=3.12 uv -y
conda activate GenD
uv pip install -r requirements.txt
```

### Formatting and Linting
Check and format before committing:
```bash
ruff check . --fix
ruff format .
```
*Note: Pyright typechecking is explicitly configured to `"off"` under `pyproject.toml`.*

### Running Experiments (Train & Test)
Experiments are configured under `src/exp/` and loaded dynamically via `run_exp.py`.
* **Train an experiment:**
  ```bash
  python run_exp.py <experiment_name>
  ```
* **Evaluate an experiment:**
  ```bash
  python run_exp.py <test_experiment_name> --from_exp <train_experiment_name> --test
  ```

### Running Local Validation/Verification Tests
The repository does not use pytest/unittest. Instead, local testing relies on custom python scripts executing specific test suite configurations from `src/exp/`:
* `python test_gend_clip.py`
* `python test_gend_pe.py`
* `python test_gend_dino.py`
* `python test_pdm_clip.py`
* `python test_pdm_xception.py`
* `python test_confdf.py`

### Launching the Web UI (Gradio App)
```bash
python app/run.py
```

---

## 📁 Repository Structure & Data Locations
* **`config/datasets/`**: Holds Git-ignored `.txt` files containing list paths of preprocessed face images relative to the workspace.
* **`datasets/`**: Local cache directory of preprocessed images for training/testing.
* **`runs/`**: Output directories for trained weights, metrics, and tensorboard logs.
