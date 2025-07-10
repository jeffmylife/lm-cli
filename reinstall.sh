#!/bin/bash

# Install or update lm-cli using uv
echo "Installing/updating lm-cli using uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv first..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the shell to make uv available
    source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
fi

# Install or update the CLI tool globally using uv
echo "Installing lm-cli globally with uv tool install..."
uv tool install --force git+https://github.com/jeffmylife/lm-cli.git

echo "Installation complete! You can now use the 'lm' command."
echo "Run 'lm --help' to get started."
