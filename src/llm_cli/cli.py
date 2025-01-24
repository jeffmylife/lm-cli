import sys
import warnings
from typing import Optional, List
import os
import base64

# Suppress Pydantic warning about config keys
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:*")

import typer
import litellm
from litellm import completion
from openai import OpenAI
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
litellm.drop_params = True


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def stream_llm_response(
    model: str,
    prompt: str,
    messages: List[dict],
    images: Optional[List[str]] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    show_reasoning: bool = False,
):
    """Stream responses from the LLM and format them using Rich."""
    try:
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
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}, *image_contents],
                    }
                ]
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

        # Initialize strings to accumulate the response
        accumulated_reasoning = ""
        accumulated_content = ""
        in_reasoning_phase = True
        has_shown_content = False

        # Use Rich's Live display for dynamic updates
        with Live(
            Text(""),
            console=console,
            refresh_per_second=10,
            vertical_overflow="visible",
            auto_refresh=True,
        ) as live:
            # For DeepSeek models with reasoning, use OpenAI client directly
            if "deepseek" in model.lower() and show_reasoning:
                client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                )
                response_stream = client.chat.completions.create(
                    model=model.split("/")[-1],  # Remove 'deepseek/' prefix
                    messages=messages,
                    stream=True,
                )

                for chunk in response_stream:
                    if chunk.choices[0].delta.reasoning_content:
                        reasoning = chunk.choices[0].delta.reasoning_content
                        accumulated_reasoning += reasoning
                        # Try to render as markdown, fallback to plain text
                        try:
                            md = Markdown(
                                "🤔 Thinking: " + accumulated_reasoning,
                                style="dim",
                            )
                            live.update(md)
                        except Exception:
                            live.update(
                                Text(
                                    "🤔 Thinking: " + accumulated_reasoning, style="dim"
                                )
                            )
                    elif chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        # Transition from reasoning to content phase
                        if in_reasoning_phase and accumulated_reasoning:
                            console.print(
                                Markdown(
                                    "🤔 Thinking: " + accumulated_reasoning, style="dim"
                                )
                            )
                            console.print()  # Add a line break between phases
                            in_reasoning_phase = False

                        accumulated_content += content
                        if not has_shown_content:
                            has_shown_content = True

                        # Try to render as markdown, fallback to plain text
                        try:
                            prefix = "💭 Response: "
                            md = Markdown(
                                prefix + accumulated_content,
                                style="markdown.text",
                                code_theme="monokai",
                                inline_code_lexer="python",
                            )
                            live.update(md)
                        except Exception:
                            prefix = "💭 Response: "
                            live.update(Text(prefix + accumulated_content))
            else:
                # Use litellm for all other models
                response_stream = completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )

                for chunk in response_stream:
                    # Extract content from the chunk
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")

                    if content:
                        accumulated_content += content
                        if not has_shown_content:
                            has_shown_content = True
                            console.print()  # Add initial line break for content

                        # Try to render as markdown, fallback to plain text
                        try:
                            md = Markdown(
                                accumulated_content,
                                style="markdown.text",
                                code_theme="monokai",
                                inline_code_lexer="python",
                            )
                            live.update(md)
                        except Exception:
                            live.update(Text(accumulated_content))

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
    context: Optional[str] = typer.Option(
        None,
        "--context",
        "-c",
        help="Path to a file to use as context for the prompt",
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", "-t", help="Maximum number of tokens to generate"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-temp", help="Sampling temperature (0.0 to 1.0)"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    think: bool = typer.Option(
        False,
        "--think",
        help="Show the model's reasoning process (only works with DeepSeek models)",
    ),
):
    """Chat with an LLM model and get markdown-formatted responses. Supports image input for compatible models."""

    print("Starting chat function...")  # Debug print

    if debug:
        print("Debug mode enabled")  # Basic print for debugging
        litellm.set_verbose = True

    # Join the prompt list into a single string
    prompt_text = " ".join(prompt)
    display_text = prompt_text

    # Prepare the message content
    message_content = prompt_text

    # If context file is provided, read it and append to both display and message
    if context:
        try:
            with open(context, "r") as f:
                context_content = f.read()
                display_text = f"{prompt_text}\n\n# {os.path.basename(context)}\n..."
                message_content = f"{prompt_text}\n\nHere's the content of {os.path.basename(context)}:\n\n{context_content}"
        except Exception as e:
            console.print(f"[red]Error reading context file: {str(e)}[/red]")
            sys.exit(1)

    # Create the messages list
    messages = [{"role": "user", "content": message_content}]

    print(f"Prompt: {display_text}")  # Debug print

    # Validate and check API keys based on the model
    model_lower = model.lower()

    if any(name in model_lower for name in ["gpt", "openai"]):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print(
                "[red]Error: OPENAI_API_KEY environment variable is not set[/red]"
            )
            sys.exit(1)
        if debug:
            console.print(
                f"[dim]Found OpenAI API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
            )
    elif any(name in model_lower for name in ["claude", "anthropic"]):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            console.print(
                "[red]Error: ANTHROPIC_API_KEY environment variable is not set[/red]"
            )
            sys.exit(1)
        if debug:
            console.print(
                f"[dim]Found Anthropic API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
            )
    elif "gemini" in model_lower:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            console.print(
                "[red]Error: GEMINI_API_KEY environment variable is not set[/red]"
            )
            sys.exit(1)
        if debug:
            console.print(
                f"[dim]Found Gemini API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
            )
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
    try:
        stream_llm_response(
            model=model,
            prompt=prompt_text,
            messages=messages,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
            show_reasoning=think,
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Basic print for errors
        if debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
