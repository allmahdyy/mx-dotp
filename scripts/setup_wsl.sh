#!/usr/bin/env bash
# One-time toolchain setup inside WSL Ubuntu 24.04 (or native Ubuntu).
#   bash scripts/setup_wsl.sh
#
# Installs Icarus Verilog, Verilator, a Python venv with cocotb, and
# OpenROAD-flow-scripts.  ORFS's dependency installer puts Yosys, OR-Tools,
# Boost, etc. under $ORFS_HOME/dependencies; OpenROAD itself is then built
# with CMake against those.  (ORFS's default Bazel build recompiles every
# dependency from source; on a 4-core laptop it was on course for >16 h,
# while the CMake build reuses the installed dependencies and is far quicker.)
#
# Needs sudo for apt and the dependency installer.  On a small WSL VM give
# it more memory first (%USERPROFILE%\.wslconfig: [wsl2] memory=6GB swap=8GB,
# then `wsl --shutdown`).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ORFS_HOME="${ORFS_HOME:-$HOME/OpenROAD-flow-scripts}"
VENV="${VENV:-$HOME/.venvs/mx-dotp}"
THREADS="${THREADS:-3}"

sudo apt-get update
sudo apt-get install -y git make g++ python3 python3-venv python3-pip iverilog verilator

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$REPO/requirements.txt"

if [ ! -d "$ORFS_HOME" ]; then
    git clone --recursive https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts "$ORFS_HOME"
fi
cd "$ORFS_HOME"
sudo ./setup.sh 2>&1 | tee "$HOME/orfs_setup.log"

. ./dev_env.sh
tools/OpenROAD/etc/Build.sh -cmake-build -no-gui -no-tests -threads="$THREADS" \
    -dir="$ORFS_HOME/tools/OpenROAD/build" \
    -prefix="$ORFS_HOME/tools/install/OpenROAD"

cat <<EOF

Done.  In each new shell:
    source $VENV/bin/activate
    source $ORFS_HOME/env.sh
    export ORFS_HOME=$ORFS_HOME
EOF
