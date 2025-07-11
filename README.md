# homebrew-streamlm

Homebrew tap for StreamLM - A command-line interface for interacting with various Large Language Models.

## Installation

```bash
brew install jeffmylife/streamlm/streamlm
```

Or add the tap first:
```bash
brew tap jeffmylife/streamlm
brew install streamlm
```

## Usage

After installation, you can use the `lm` command:

```bash
lm "explain quantum computing"
lm -m gpt-4o "write a Python function"
lm -m claude-3-5-sonnet "analyze this data"
```

## About

StreamLM provides a beautiful command-line interface for interacting with various Large Language Models including:

- **OpenAI**: GPT-4o, o1, o3-mini, GPT-4o-mini
- **Anthropic**: Claude-3-7-sonnet, Claude-3-5-sonnet, Claude-3-5-haiku
- **Google**: Gemini-2.5-flash, Gemini-2.5-pro, Gemini-2.0-flash-thinking
- **DeepSeek**: DeepSeek-R1, DeepSeek-V3
- **xAI**: Grok-4, Grok-3-beta, Grok-3-mini-beta
- **Local models**: Via Ollama (Llama3.3, Qwen2.5, DeepSeek-Coder, etc.)

## Links

- [Main Repository](https://github.com/jeffmylife/streamlm)
- [PyPI Package](https://pypi.org/project/streamlm/)
- [Documentation](https://github.com/jeffmylife/streamlm#readme)
