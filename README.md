# LM CLI

A command-line interface for interacting with various LLM models (GPT, Claude, Gemini, etc.) with streaming output.

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/jeffmylife/lm-cli/master/reinstall.sh | bash
```

Or install manually:
```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the CLI
uv tool install git+https://github.com/jeffmylife/lm-cli.git
```

## Setup

Set your API keys:
```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
export GEMINI_API_KEY=your_key_here
export DEEPSEEK_API_KEY=your_key_here
export OPENROUTER_API_KEY=your_key_here
```

For local models, install Ollama:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull llama2
```

## Usage

```bash
# Basic usage
lm tell me a joke about programming

# With specific model
lm --model gpt-4o explain quantum computing

# With reasoning models
lm --model deepseek/deepseek-reasoner --think "solve this step by step: 2+2*3"

# With images
lm --image photo.jpg describe this image

# With context files
lm --context src/main.py explain this code

# Chain commands
lm "write a haiku about coding" | lm "make it funnier"
```

## Models

- **OpenAI**: `gpt-4o`, `gpt-3.5-turbo`, `o1-preview`, `o1-mini`
- **Anthropic**: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- **Google**: `gemini/gemini-2.5-flash`, `gemini/gemini-2.0-flash`
- **DeepSeek**: `deepseek/deepseek-reasoner`, `deepseek/deepseek-coder`
- **OpenRouter**: `openrouter/openai/gpt-4o`, `openrouter/anthropic/claude-3-sonnet`
- **Ollama**: `ollama/llama2`, `ollama/mistral`, `ollama/codellama`

## Options

- `--model`, `-m`: Choose model (default: `gemini/gemini-2.5-flash`)
- `--max-tokens`, `-t`: Max tokens to generate
- `--temperature`, `-temp`: Sampling temperature (0.0-1.0)
- `--context`, `-c`: Path to context file
- `--image`, `-i`: Path to image file (can use multiple times)
- `--think`: Show reasoning process (for reasoning models)
- `--debug`, `-d`: Enable debug mode

## Development

```bash
# Clone and run locally
git clone https://github.com/jeffmylife/lm-cli.git
cd lm-cli
uv run lm hello world
```