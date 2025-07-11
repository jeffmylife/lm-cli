#!/bin/bash

# Install or update streamlm using the best available method
echo "Installing/updating streamlm..."

# Function to install with uv
install_with_uv() {
    echo "Installing with uv..."
    if ! command -v uv &> /dev/null; then
        echo "uv is not installed. Installing uv first..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Source the shell to make uv available
        source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
    fi
    
    # Try PyPI first, fall back to GitHub
    if uv tool install --force streamlm; then
        echo "✅ Installed from PyPI"
    else
        echo "PyPI installation failed, trying GitHub..."
        uv tool install --force git+https://github.com/jeffmylife/streamlm.git
        echo "✅ Installed from GitHub"
    fi
}

# Function to install with pipx
install_with_pipx() {
    echo "Installing with pipx..."
    if ! command -v pipx &> /dev/null; then
        echo "pipx is not installed. Installing pipx first..."
        python -m pip install --user pipx
        python -m pipx ensurepath
    fi
    
    # Try PyPI first, fall back to GitHub
    if pipx install --force streamlm; then
        echo "✅ Installed from PyPI"
    else
        echo "PyPI installation failed, trying GitHub..."
        pipx install --force git+https://github.com/jeffmylife/streamlm.git
        echo "✅ Installed from GitHub"
    fi
}

# Function to install with pip
install_with_pip() {
    echo "Installing with pip..."
    # Try PyPI first, fall back to GitHub
    if pip install --upgrade streamlm; then
        echo "✅ Installed from PyPI"
    else
        echo "PyPI installation failed, trying GitHub..."
        pip install --upgrade git+https://github.com/jeffmylife/streamlm.git
        echo "✅ Installed from GitHub"
    fi
}

# Check for Homebrew first (macOS)
if command -v brew &> /dev/null; then
    echo "Homebrew detected. You can also install with:"
    echo "  brew install jeffmylife/streamlm/streamlm"
    echo ""
fi

# Try installation methods in order of preference
if command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; then
    install_with_uv
elif command -v pipx &> /dev/null || python -m pip install --user pipx; then
    install_with_pipx
elif command -v pip &> /dev/null; then
    install_with_pip
else
    echo "❌ No suitable Python package manager found."
    echo "Please install Python and pip, then run this script again."
    exit 1
fi

echo ""
echo "🎉 Installation complete! You can now use the 'lm' command."
echo "Run 'lm --help' to get started."
echo ""
echo "Set your API keys:"
echo "  export OPENAI_API_KEY=your_key_here"
echo "  export ANTHROPIC_API_KEY=your_key_here"
echo "  export GEMINI_API_KEY=your_key_here"
