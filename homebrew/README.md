# homebrew-streamlm

Homebrew formula for StreamLM - A command-line interface for interacting with various Large Language Models.

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

## Formula Details

The formula installs the `streamlm` Python package and its dependencies in an isolated environment, making the `lm` command available system-wide.

### Dependencies

- Python 3.12+
- Various Python packages (installed automatically)

### Build Process

The formula uses Python's virtualenv to create an isolated environment and installs all dependencies automatically.

## Development

To update the formula:

```bash
# Manual update
gh workflow run update-homebrew.yml -f version=0.1.3

# Automatic update happens on each GitHub release
```

## Testing

Test the formula locally:

```bash
brew install --build-from-source homebrew/Formula/streamlm.rb
lm --version
lm --help
``` 