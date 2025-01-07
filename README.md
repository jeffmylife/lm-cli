# LLM CLI

A command-line interface for interacting with various LLM models (GPT, Claude, etc.) with markdown-formatted streaming output.

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd llm-cli
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys as environment variables:
```bash
# For OpenAI models (GPT-4, GPT-3.5)
export OPENAI_API_KEY=your_key_here

# For Anthropic models (Claude)
export ANTHROPIC_API_KEY=your_key_here
```

## Usage

Basic usage:
```bash
python src/cli.py "Your prompt here"
```

With options:
```bash
python src/cli.py --model gpt-4-turbo-preview --temperature 0.8 "Your prompt here"
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

### Examples

1. Using GPT-4 Turbo (default):
```bash
python src/cli.py "Write a technical blog post about async/await in Python"
```

2. Using Claude 3 Opus for complex tasks:
```bash
python src/cli.py --model claude-3-opus "Explain quantum computing with detailed analogies"
```

3. Using a faster model for simple tasks:
```bash
python src/cli.py --model claude-3-haiku "Write a quick product description"
```

## Features

- Streaming responses in real-time
- Markdown formatting for better readability
- Fallback to plain text when markdown parsing fails
- Simple command-line interface
- Configurable temperature and max tokens
- Support for multiple LLM providers
- Model validation and helpful suggestions
- Smart error handling for API keys 