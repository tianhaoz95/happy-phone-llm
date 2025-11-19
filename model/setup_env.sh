#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

VENV_DIR=".venv"

echo "Setting up Python virtual environment..."

# Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created in $VENV_DIR"
else
    echo "Virtual environment already exists in $VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated."

# Install dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install -r model/requirements.txt

echo "Python environment setup complete."
echo "To activate the environment, run: source $VENV_DIR/bin/activate"
echo "To deactivate, run: deactivate"
