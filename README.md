# LM CLI

A command-line interface for interacting with various LLM models (GPT, Claude, etc.) with markdown-formatted streaming output.

## Prerequisites

- For OpenAI models: An OpenAI API key
- For Anthropic models: An Anthropic API key
- For Ollama models: [Ollama](https://ollama.ai) installed and running

## Installation

### Option 1: Global Installation (Recommended)

1. Install pipx if you haven't already:
```bash
# On macOS
brew install pipx
pipx ensurepath

# On Linux
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

2. Install the CLI globally:
```bash
pipx install git+https://github.com/jeffmylife/lm-cli.git
```

This will make the `lm` command available everywhere for your user account.

### Option 2: Development Installation

1. Clone this repository:
```bash
git clone https://github.com/jeffmylife/lm-cli.git
cd lm-cli
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install in development mode:
```bash
pip install -e .
```

### API Keys and Service Setup

1. Set up API keys in your `~/.zshrc` (or equivalent):
```bash
# For OpenAI models (GPT-4, GPT-3.5)
export OPENAI_API_KEY=your_key_here

# For Anthropic models (Claude)
export ANTHROPIC_API_KEY=your_key_here

# For Google models (Gemini)
export GEMINI_API_KEY=your_key_here
```

2. For Ollama models:
```bash
# Install Ollama
brew install ollama

# Start the Ollama server
ollama serve

# Pull models you want to use
ollama pull llama2
ollama pull mistral
ollama pull codellama
ollama pull gemma
```

Then reload your shell:
```bash
source ~/.zshrc
```

## Usage

After installation, you can use the `lm` command from anywhere. Just type your prompt directly - no quotes needed!

Basic usage:
```bash
lm tell me a joke about programming
```

With options:
```bash
lm --model gpt-4-turbo-preview --temperature 0.8 write me a haiku about coding
```

### Available Options

- `--model`, `-m`: Choose the LLM model (default: "gpt-4-turbo-preview")
- `--max-tokens`, `-t`: Maximum number of tokens to generate
- `--temperature`, `-temp`: Sampling temperature (0.0 to 1.0, default: 0.7)

### Supported Models

#### OpenAI Models
- `gpt-4-turbo-preview` (default, recommended for most uses)
- `gpt-4` (more stable, might be slower)
- `gpt-3.5-turbo` (faster, more cost-effective)

#### Anthropic Models
- `claude-3-opus` (most capable)
- `claude-3-sonnet` (balanced performance)
- `claude-3-haiku` (fastest, most cost-effective)

#### Ollama Models (Local)
- `ollama/llama2` (general purpose)
- `ollama/mistral` (efficient, good performance)
- `ollama/codellama` (code-focused)
- `ollama/gemma` (Google's latest model)

### Examples

1. Using GPT-4 Turbo (default):
```bash
lm explain how async/await works in Python with examples
```

2. Using Claude 3 Opus for complex tasks:
```bash
lm --model claude-3-opus explain quantum computing with detailed analogies
```

3. Using Ollama's Mistral locally:
```bash
lm --model ollama/mistral write a quick product description for a coffee mug
```

4. Using CodeLlama for programming tasks:
```bash
lm --model ollama/codellama write a Python function to calculate Fibonacci numbers
```

## Features

- Streaming responses in real-time
- Markdown formatting for better readability
- Fallback to plain text when markdown parsing fails
- Simple command-line interface
- No quotes needed - just type your prompt!
- Configurable temperature and max tokens
- Support for multiple LLM providers:
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Ollama (Local models)
- Model validation and helpful suggestions
- Smart error handling for API keys and services

## Development

To contribute or modify the CLI:

1. Clone the repository
2. Install in development mode: `pip install -e .`
3. Make your changes
4. Test your changes: `lm hello world`

## TODO

- [ ] support chat feature
- [ ] support system prompt
- [ ] support no prompt (for images)
- [x] support input images