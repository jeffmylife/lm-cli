import sys
import warnings
from typing import Optional, List
import os
import base64
from pathlib import Path

# Suppress Pydantic warning about config keys
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:*")

import typer
import litellm
from litellm import completion
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from rich.traceback import install

install()

# Initialize Typer app and Rich console
app = typer.Typer(help="A CLI tool for interacting with various LLMs", name="llm")
console = Console()

litellm.suppress_debug_info = True


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def stream_llm_response(
    model: str,
    prompt: str,
    images: Optional[List[str]] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
):
    """Stream responses from the LLM and format them using Rich."""
    try:
        # Prepare the messages
        messages = []

        # Add images if provided
        if images:
            # For models that expect base64
            if any(
                name in model.lower() for name in ["gpt-4-vision", "gemini", "claude-3"]
            ):
                image_contents = []
                for img_path in images:
                    if img_path.startswith(("http://", "https://")):
                        image_contents.append(
                            {"type": "image_url", "image_url": img_path}
                        )
                    else:
                        base64_image = encode_image_to_base64(img_path)
                        image_contents.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        )
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}, *image_contents],
                    }
                )
            # For Ollama vision models
            elif "ollama" in model.lower():
                for img_path in images:
                    if img_path.startswith(("http://", "https://")):
                        console.print(
                            "[red]Error: Ollama vision models only support local image files[/red]"
                        )
                        sys.exit(1)
                    else:
                        base64_image = encode_image_to_base64(img_path)
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ]
            else:
                console.print(
                    "[red]Error: This model doesn't support image input[/red]"
                )
                sys.exit(1)
        else:
            messages.append({"role": "user", "content": prompt})

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

        # Initialize an empty string to accumulate the response
        accumulated_text = ""

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
        "gemini/gemini-2.0-flash-exp",
        "--model",
        "-m",
        help="The LLM model to use. Examples: gpt-4-turbo-preview, claude-3-opus, ollama/llama2",
    ),
    images: Optional[List[str]] = typer.Option(
        None,
        "--image",
        "-i",
        help="Path to image file or URL. Can be specified multiple times for multiple images.",
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", "-t", help="Maximum number of tokens to generate"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-temp", help="Sampling temperature (0.0 to 1.0)"
    ),
):
    """Chat with an LLM model and get markdown-formatted responses. Supports image input for compatible models."""

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
    elif "gemini" in model_lower:
        if not os.getenv("GEMINI_API_KEY"):
            console.print(
                "[red]Error: GEMINI_API_KEY environment variable is not set[/red]"
            )
            sys.exit(1)
        # Format for litellm's Gemini support (using API key instead of Vertex)
        # model = model.replace("gemini/", "google/")
    elif "ollama" in model_lower:
        # Check if Ollama server is running
        try:
            import requests

            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                console.print(
                    "[red]Error: Ollama server is not running. Please start it with 'ollama serve'[/red]"
                )
                sys.exit(1)
        except requests.exceptions.ConnectionError:
            console.print(
                "[red]Error: Cannot connect to Ollama server. Please start it with 'ollama serve'[/red]"
            )
            sys.exit(1)

    # Show what model we're using
    console.print(f"[dim]Using model: {model}[/dim]")
    if images:
        console.print(
            f"[dim]With {len(images)} image{'s' if len(images) > 1 else ''}[/dim]"
        )
    console.print()  # Add a blank line for cleaner output

    # Configure model-specific settings
    if "ollama" in model_lower:
        litellm.set_verbose = False
        os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
        # Format for litellm's Ollama support
        model = f"ollama/{model.split('/')[-1]}"

    # Stream the response
    stream_llm_response(
        model=model,
        prompt=prompt_text,
        images=images,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def main():
    """Entry point for the CLI."""
    app()
