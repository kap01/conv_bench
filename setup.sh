#!/bin/bash
# Build the Python environment for the convolution benchmark on gpu01.dice.priv.
# Project lives on gpu01's local disk (home is quota-limited): /storage/kp14102/conv_bench,
# symlinked from ~/conv_bench.  Base interpreter: Python 3.12 from ~/bkstautau/env.
set -euo pipefail
P=/storage/kp14102/conv_bench
mkdir -p "$P/plots" "$P/results"
ln -sfn "$P" ~/conv_bench
cd "$P"
[ -x venv/bin/python ] || ~/bkstautau/env/bin/python -m venv venv
./venv/bin/pip install --no-cache-dir -q --upgrade pip
./venv/bin/pip install --no-cache-dir -q "tensorflow[and-cuda]" scipy matplotlib pandas tabulate jupyter ipykernel
./venv/bin/python -m ipykernel install --user --name conv_bench --display-name "Python 3.12 (conv_bench)"
./venv/bin/python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
