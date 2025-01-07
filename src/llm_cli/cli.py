#!/usr/bin/env python3

import os
import sys
import warnings
from typing import Optional

# Suppress Pydantic warning about config keys
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:*")

import typer
import litellm
from litellm import completion
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

# Initialize Typer app and Rich console
app = typer.Typer(help="A CLI tool for interacting with various LLMs", name="llm")
console = Console()

litellm.suppress_debug_info = True


def stream_llm_response(
    model: str,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
):
    """Stream responses from the LLM and format them using Rich."""
    try:
        # Prepare the messages
        messages = [{"role": "user", "content": prompt}]

        # Initialize an empty string to accumulate the response
        accumulated_text = ""

        # Create the completion with streaming
        response_stream = completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        # Configure console for clean output
        console.width = min(console.width, 100)  # Limit width for readability

        # Use Rich's Live display for dynamic updates
        with Live(
            Text(""),
            console=console,
            refresh_per_second=10,
            vertical_overflow="visible",
            auto_refresh=True,
        ) as live:
            for chunk in response_stream:
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    accumulated_text += content
                    # Try to render as markdown, fallback to plain text if it fails
                    try:
                        md = Markdown(
                            accumulated_text,
                            style="markdown.text",
                            code_theme="monokai",
                            inline_code_lexer="python",
                        )
                        live.update(md)
                    except Exception:
                        live.update(Text(accumulated_text))

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@app.command(context_settings={"ignore_unknown_options": True})
def chat(
    prompt: list[str] = typer.Argument(..., help="The prompt to send to the LLM"),
    model: str = typer.Option(
        "gemini/gemini-1.5-flash-latest",
        "--model",
        "-m",
        help="The LLM model to use. Examples: gpt-4-turbo-preview, gpt-3.5-turbo, claude-3-opus, claude-3-sonnet, claude-3-haiku",
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", "-t", help="Maximum number of tokens to generate"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-temp", help="Sampling temperature (0.0 to 1.0)"
    ),
):
    """Chat with an LLM model and get markdown-formatted responses."""

    # Join the prompt list into a single string
    prompt_text = " ".join(prompt)

    # Validate and check API keys based on the model
    model_lower = model.lower()

    if any(name in model_lower for name in ["gpt", "openai"]):
        if not os.getenv("OPENAI_API_KEY"):
            console.print(
                "[red]Error: OPENAI_API_KEY environment variable is not set[/red]"
            )
            sys.exit(1)
    elif any(name in model_lower for name in ["claude", "anthropic"]):
        if not os.getenv("ANTHROPIC_API_KEY"):
            console.print(
                "[red]Error: ANTHROPIC_API_KEY environment variable is not set[/red]"
            )
            sys.exit(1)

    # Show what model we're using
    console.print(f"[dim]Using model: {model}[/dim]")
    console.print()  # Add a blank line for cleaner output

    # Stream the response
    stream_llm_response(
        model=model,
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def main():
    """Entry point for the CLI."""
    app()
