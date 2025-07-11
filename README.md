# StreamLM

[![PyPI](https://img.shields.io/pypi/v/streamlm.svg)](https://pypi.org/project/streamlm/)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/jeffmylife/streamlm.svg)](https://github.com/jeffmylife/streamlm/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jeffmylife/streamlm/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/streamlm.svg)](https://pypi.org/project/streamlm/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A command-line interface for interacting with various Large Language Models with beautiful streaming markdown output.

## Quick Start

First, install StreamLM using `pip`, `pipx`, `uv`, or Homebrew:

```bash
# Using pip
pip install streamlm

# Using pipx (recommended for CLI tools)
pipx install streamlm

# Using uv (fastest)
uv tool install streamlm

# Using Homebrew
brew install jeffmylife/streamlm/streamlm
```

If you have an [OpenAI API key](https://platform.openai.com/api-keys), you can start using it immediately:

```bash
export OPENAI_API_KEY=your_key_here
lm "Ten fun names for a pet pelican"
```

## Installation Methods

### PyPI (Recommended)

The easiest way to install StreamLM is from PyPI:

```bash
# Install globally with pip
pip install streamlm

# Install in isolated environment with pipx (recommended)
pipx install streamlm

# Install with uv (fastest)
uv tool install streamlm
```

### Homebrew

If you prefer Homebrew:

```bash
# Add the tap and install
brew install jeffmylife/streamlm/streamlm

# Or install directly from the tap
brew tap jeffmylife/streamlm
brew install streamlm
```

### Development Installation

For development or to get the latest features:

```bash
# Clone and install in development mode
git clone https://github.com/jeffmylife/streamlm.git
cd streamlm
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/jeffmylife/streamlm.git

# Using uv for development
uv tool install git+https://github.com/jeffmylife/streamlm.git
```

### One-liner Installation Script

For quick installation with automatic setup:

```bash
curl -fsSL https://raw.githubusercontent.com/jeffmylife/streamlm/master/reinstall.sh | bash
```

## Setup

Set your API keys:
```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
export GEMINI_API_KEY=your_key_here
export DEEPSEEK_API_KEY=your_key_here
export XAI_API_KEY=your_key_here
export OPENROUTER_API_KEY=your_key_here
```

For local models, install Ollama:
```bash
# linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull llama3.3
```

```bash
# macos
brew install ollama
ollama serve
ollama pull llama2
```


## Usage

### Quick Examples

```bash
# Use latest Gemini (default)
lm "explain quantum computing"

# Use specific models
lm -m gpt-4o "write a Python function to sort a list"
lm -m claude-3-7-sonnet "analyze this data trend"
lm -m xai/grok-4 "solve this math problem step by step"

# Local models
lm -m ollama/llama3.3 "help me debug this code"
```

### Available Models

**OpenAI (Latest)**
- `gpt-4o` - GPT-4 Omni multimodal model
- `o1` - Advanced reasoning model
- `o3-mini` - Efficient reasoning model
- `gpt-4o-mini` - Fast, cost-effective model

**Anthropic (Latest)**
- `claude-3-7-sonnet` - Hybrid reasoning with extended thinking
- `claude-3-5-sonnet` - Balanced performance and speed
- `claude-3-5-haiku` - Ultra-fast responses

**Google (Latest)**
- `gemini-2.5-flash` - Ultra-fast streaming (default)
- `gemini-2.5-pro` - Advanced reasoning and multimodal
- `gemini-2.0-flash-thinking` - Reasoning with visible thoughts

**DeepSeek (Latest)**
- `deepseek-r1` - Advanced reasoning at low cost
- `deepseek-v3` - High-performance general model

**xAI (Latest)**
- `xai/grok-4` - Most advanced reasoning model
- `xai/grok-3-beta` - High-performance reasoning
- `xai/grok-3-mini-beta` - Fast reasoning model

**Mistral (Latest)**
- `mistral-large-3` - Flagship performance model
- `mistral-small-3.1` - Efficient 24B parameter model
- `pixtral-large` - 124B multimodal model

**Local (via Ollama)**
- `ollama/llama3.3` - Meta's latest 70B model
- `ollama/qwen2.5` - Alibaba's multilingual model
- `ollama/deepseek-coder` - Code-specialized model

### Advanced Usage

```bash
# Pipe input
echo "Explain this code" | lm -m claude-3-7-sonnet

# File input
lm -m gemini-2.5-pro < document.txt

# Reasoning models with visible thinking
lm -m xai/grok-4 --think "solve this complex problem"
lm -m deepseek-r1 --think "analyze this step by step"
```

## Development

```bash
git clone https://github.com/jeffmylife/streamlm.git
cd streamlm
uv run lm hello world
```