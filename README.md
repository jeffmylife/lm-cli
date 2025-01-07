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
# For OpenAI models (GPT-3.5, GPT-4)
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
python src/cli.py --model gpt-4 --temperature 0.8 "Your prompt here"
```

### Available Options

- `--model`, `-m`: Choose the LLM model (default: "gpt-3.5-turbo")
- `--max-tokens`, `-t`: Maximum number of tokens to generate
- `--temperature`, `-temp`: Sampling temperature (0.0 to 1.0, default: 0.7)

### Examples

1. Using GPT-3.5 Turbo (default):
```bash
python src/cli.py "Write a haiku about programming"
```

2. Using GPT-4 with custom temperature:
```bash
python src/cli.py --model gpt-4 --temperature 0.9 "Explain quantum computing"
```

3. Using Claude:
```bash
python src/cli.py --model claude-2 "Write a short story"
```

## Supported Models

The CLI supports any model available through LiteLLM, including:
- OpenAI models (gpt-3.5-turbo, gpt-4, etc.)
- Anthropic models (claude-2, claude-instant, etc.)
- And more through LiteLLM's supported providers

## Features

- Streaming responses in real-time
- Markdown formatting for better readability
- Fallback to plain text when markdown parsing fails
- Simple command-line interface
- Configurable temperature and max tokens
- Support for multiple LLM providers 