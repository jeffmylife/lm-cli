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

The formula uses Python's virtualenv to create an isolated environment and installs all required dependencies automatically.

## Development

To update the formula:

1. Update the version number and SHA256 hash
2. Test the formula locally
3. Submit a pull request

## Links

- [Main Repository](https://github.com/jeffmylife/streamlm)
- [PyPI Package](https://pypi.org/project/streamlm/)
- [Documentation](https://github.com/jeffmylife/streamlm#readme) 