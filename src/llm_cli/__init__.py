"""LLM CLI - A command-line interface for interacting with various LLM models."""

try:
    from importlib.metadata import version

    __version__ = version("streamlm")
except ImportError:
    # Fallback for older Python versions or development installs
    __version__ = "0.1.2"
