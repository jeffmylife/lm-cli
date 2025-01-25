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

- `--model`, `-m`: Choose the LLM model (default: "gemini/gemini-2.0-flash-exp")
- `--max-tokens`, `-t`: Maximum number of tokens to generate
- `--temperature`, `-temp`: Sampling temperature (0.0 to 1.0, default: 0.7)
- `--context`, `-c`: Path to a file to use as context for the prompt
- `--think`: Show the model's reasoning process (only works with DeepSeek models)

### Supported Models

#### OpenAI Models
- `gpt-4o` 
- `gpt-3.5-turbo` (faster, more cost-effective)

#### Anthropic Models
- `claude-3-opus-20240229` (most capable)
- `claude-3-sonnet-20240229` (balanced performance)
- `claude-3-haiku-20240307` (fastest, most cost-effective)

#### Google Models
- `gemini/gemini-2.0-flash-exp` (default, fast and capable)
- `gemini/gemini-pro` (standard model)

#### Ollama Models (Local)
- `ollama/llama2` (general purpose)
- `ollama/mistral` (efficient, good performance)
- `ollama/codellama` (code-focused)
- `ollama/gemma` (Google's latest model)

#### DeepSeek Models
- `deepseek/deepseek-reasoner` (shows reasoning process with --think flag)
- `deepseek/deepseek-coder` (specialized for code)

### Examples

1. Using GPT-4o (default):
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

5. Explaining code with context:
```bash
lm --context src/main.py explain this code
```

6. Using DeepSeek Reasoner with visible thinking process:
```bash
lm --model deepseek/deepseek-reasoner --think "solve this puzzle: if you have 9 coins and one is fake (lighter), how can you find it with just 2 weighings?"
```

7. Combining features:
```bash
lm --model deepseek/deepseek-reasoner --think --context src/complex.py "explain what this code does and how it could be improved"
```

8. Chaining commands with pipes:
```bash
# Ask follow-up questions about previous responses
lm "what is the capital of France?" | lm "what's the population there?"

# Build on previous responses
lm "write a short story about a cat" | lm "make this story funnier"

# Use previous output as context
echo "The sky is blue because of Rayleigh scattering" | lm "explain this in simpler terms"
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
  - DeepSeek (with visible reasoning process)
- Model validation and helpful suggestions
- Smart error handling for API keys and services
- File context support for code and text analysis
- Visible thinking process with DeepSeek models
- Command chaining with Unix pipes

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