# LM CLI

A command-line interface for interacting with various LLM models with streaming output.

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
git clone https://github.com/jeffmylife/lm-cli.git
cd lm-cli
uv run lm hello world
```