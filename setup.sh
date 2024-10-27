#!/bin/bash

# Virtual environment name
VENV_DIR="venv"

# Dependencies to install
DEPENDENCIES=("transformers[torch]" "torch" "scikit-learn" "pandas" "fastapi" "uvicorn" "accelerate")

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it before proceeding."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment in $VENV_DIR..."
python3 -m venv $VENV_DIR

# Detect operating system and activate virtual environment
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux/Ubuntu
    source $VENV_DIR/bin/activate
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows (Git Bash or Cygwin)
    source $VENV_DIR/Scripts/activate
elif [[ "$OSTYPE" == "win32" ]]; then
    # Windows Command Prompt
    .\\$VENV_DIR\\Scripts\\activate
else
    echo "Operating system not supported by this script."
    exit 1
fi

# Confirm that we are in the virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment activated successfully."
else
    echo "Failed to activate virtual environment."
    exit 1
fi

# Install dependencies within the virtual environment
echo "Installing dependencies..."
for package in "${DEPENDENCIES[@]}"; do
    pip install "$package"
done

# Save dependencies to requirements.txt
pip freeze > requirements.txt
echo "Dependencies installed and saved to requirements.txt"

echo "Setup complete! The virtual environment is ready."
