#!/usr/bin/env bash
###############################################################################
#  JointTagger – Conda + uv launcher (POSIX shell version)
#  • Creates env “jointagger” on first run
#  • run_local.sh --update   → re-solve environment.yml
#  • Activates env in THIS shell, installs uv + wheels, runs app.py
###############################################################################

# ---------- CONFIG ----------------------------------------------------------
CONDA_ROOT="${HOME}/Miniconda3"          # change only if your Miniconda lives elsewhere
ENV_NAME="jointagger"
PY_VER="3.11.9"
# ----------------------------------------------------------------------------

set -euo pipefail
DO_UPDATE=0
[[ "${1:-}" == "--update" ]] && DO_UPDATE=1

# 1) Ensure Conda is present --------------------------------------------------
if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    echo "➜  Installing Miniconda to ${CONDA_ROOT}..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    if [[ "$(uname)" == "Darwin" ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    fi
    curl -L "$MINICONDA_URL" -o /tmp/miniconda.sh || {
        echo "✖  Failed to download Miniconda";
        exit 1;
    }
    bash /tmp/miniconda.sh -b -p "${CONDA_ROOT}"
    rm -f /tmp/miniconda.sh
fi

# 2) Make Conda functions available in this shell ----------------------------
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# 3) Create env ONLY if its directory doesn’t exist --------------------------
if ! conda info --envs | grep -qx "${ENV_NAME}"; then
    echo "➜  Creating Conda env “${ENV_NAME}”…"
    if [[ -f environment.yml ]]; then
        conda env create -n "${ENV_NAME}" -f environment.yml
    else
        conda create -y -n "${ENV_NAME}" "python=${PY_VER}" "conda-forge::pytorch-gpu"
    fi
fi

# 4) Optional --update  → conda env update -----------------------------------
if [[ "${DO_UPDATE}" -eq 1 ]]; then
    echo "➜  Updating Conda env “${ENV_NAME}”…"
    if [[ -f environment.yml ]]; then
        conda env update -n "${ENV_NAME}" -f environment.yml --prune
    else
        echo "✖  No environment.yml found; skipping update."
    fi
fi

# 5) Activate env in THIS shell ----------------------------------------------
conda activate "${ENV_NAME}"

# 6) Launch the app -----------------------------------------------------------
echo "➜  Launching JointTagger …"
python app.py
