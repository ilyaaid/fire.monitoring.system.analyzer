#!/bin/bash

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment..."
  python3 -m venv $VENV_DIR
fi

echo "[INFO] Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "[INFO] Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[DONE] Setup complete. Virtual environment is ready."
