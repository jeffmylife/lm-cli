# homebrew-lm-cli

Homebrew formula for installing LM CLI - a command-line interface for interacting with various Large Language Models.

## Installation

```bash
# Add the tap and install
brew install jeffmylife/lm-cli/lm-cli

# Or install directly from the tap
brew tap jeffmylife/lm-cli
brew install lm-cli
```

## About

LM CLI is a command-line interface for interacting with various Large Language Models with beautiful streaming markdown output. It supports:

- **OpenAI models**: GPT-4o, o1, o3-mini, GPT-4o-mini
- **Anthropic models**: Claude-3-7-sonnet, Claude-3-5-sonnet, Claude-3-5-haiku
- **Google models**: Gemini-2.5-flash, Gemini-2.5-pro, Gemini-2.0-flash-thinking
- **DeepSeek models**: DeepSeek-r1, DeepSeek-v3
- **xAI models**: Grok-4, Grok-3-beta, Grok-3-mini-beta
- **Mistral models**: Mistral-large-3, Mistral-small-3.1, Pixtral-large
- **Local models**: Via Ollama (Llama3.3, Qwen2.5, DeepSeek-coder)

## Example Usage

```bash
# Use latest Gemini (default)
lm "explain quantum computing"

# Use specific models
lm -m gpt-4o "write a Python function to sort a list"
lm -m claude-3-7-sonnet "analyze this data trend"

# Local models
lm -m ollama/llama3.3 "help me debug this code"
```

## Links

- [GitHub Repository](https://github.com/jeffmylife/lm-cli)
- [PyPI Package](https://pypi.org/project/lm-cli/)
- [Documentation](https://github.com/jeffmylife/lm-cli#readme) 