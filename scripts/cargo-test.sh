#!/usr/bin/env bash
# Runs the Rust unit tests (`cargo test`).
#
# The test binary links against libpython. Pythons installed by uv keep
# libpython outside the system library paths, so we point the loader at the
# project's Python (and tell PyO3 to build against that same Python).
set -euo pipefail

PYO3_PYTHON="$(uv run --frozen python -c 'import sys; print(sys.executable)')"
python_lib_dir="$(uv run --frozen python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"

export PYO3_PYTHON
export LD_LIBRARY_PATH="$python_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"      # Linux
export DYLD_LIBRARY_PATH="$python_lib_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"  # macOS

exec cargo test "$@"
