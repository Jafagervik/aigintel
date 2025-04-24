#!/bin/bash

# Exit on any error
set -e

# Repository URL (replace with your actual repo URL)
REPO_URL="<repository-url>"
REPO_NAME=$(basename "$REPO_URL" .git)

# Step 1: Clone the repository
echo "Cloning the repository..."
if [ -d "$REPO_NAME" ]; then
    echo "Directory $REPO_NAME already exists. Pulling latest changes..."
    cd "$REPO_NAME"
    git pull
    cd ..
else
    git clone "$REPO_URL"
fi
cd "$REPO_NAME"

# Step 2: Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Ensure uv is in PATH (if installed in this session)
export PATH="$HOME/.cargo/bin:$PATH"

# Step 3: Create and activate virtual environment
echo "Creating virtual environment..."
uv venv

echo "Activating virtual environment..."
source .venv/bin/activate

# Step 4: Install dependencies
echo "Installing dependencies with uv..."
uv sync

echo "Installation complete! Virtual environment is active."
echo "To deactivate the virtual environment, run: deactivate"