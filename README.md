# LLM CLI

A command-line interface for interacting with various LLM models (GPT, Claude, etc.) with markdown-formatted streaming output.

## Installation

### Option 1: Install from source (Development)

1. Clone this repository:
```bash
git clone <your-repo-url>
cd llm-cli
```

2. Install in development mode:
```bash
pip install -e .
```

### Option 2: Install directly (User)

```bash
pip install git+https://github.com/yourusername/llm-cli.git
```

### API Keys Setup

Set up your API keys as environment variables:
```bash
# For OpenAI models (GPT-4, GPT-3.5)
export OPENAI_API_KEY=your_key_here

# For Anthropic models (Claude)
export ANTHROPIC_API_KEY=your_key_here
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

### Examples

1. Using GPT-4 Turbo (default):
```bash
lm explain how async/await works in Python with examples
```

2. Using Claude 3 Opus for complex tasks:
```bash
lm --model claude-3-opus explain quantum computing with detailed analogies
```

3. Using a faster model for simple tasks:
```bash
lm --model claude-3-haiku write a quick product description for a coffee mug
```

## Features

- Streaming responses in real-time
- Markdown formatting for better readability
- Fallback to plain text when markdown parsing fails
- Simple command-line interface
- No quotes needed - just type your prompt!
- Configurable temperature and max tokens
- Support for multiple LLM providers
- Model validation and helpful suggestions
- Smart error handling for API keys

## Development

To contribute or modify the CLI:

1. Clone the repository
2. Install in development mode: `pip install -e .`
3. Make your changes
4. Test your changes: `lm hello world` 