import sys
import warnings
from typing import Optional, List, cast, Any
import os
import base64
import subprocess
import time

# Suppress Pydantic warning about config keys
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:*")

import typer
import litellm
from litellm import completion
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console
from rich.traceback import install
from .streaming_markdown import StreamingMarkdownRenderer

install()

# Initialize Typer app and Rich console
app = typer.Typer(help="A CLI tool for interacting with various LLMs", name="llm")
console = Console()


def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short hash
        else:
            return "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return "unknown"


def version_callback(value: bool):
    """Callback for --version option."""
    if value:
        from . import __version__

        commit_hash = get_git_commit_hash()
        console.print(f"streamlm version {__version__} (commit: {commit_hash})")
        raise typer.Exit()


litellm.suppress_debug_info = True
litellm.drop_params = True


def is_reasoning_model(model: str) -> bool:
    """Check if a model supports reasoning/thinking capabilities."""
    model_lower = model.lower()

    # DeepSeek reasoning models
    if any(name in model_lower for name in ["deepseek-reasoner", "deepseek-r1"]):
        return True

    # OpenAI o1 models
    if any(name in model_lower for name in ["o1-preview", "o1-mini", "o1-pro"]):
        return True

    # xAI Grok models with reasoning
    if any(name in model_lower for name in ["grok-3", "grok-4"]):
        return True

    # Add other reasoning models as they become available
    # Example: if "reasoning" in model_lower or "think" in model_lower:
    #     return True

    return False


def get_model_provider(model: str) -> str:
    """Determine the provider for a given model."""
    model_lower = model.lower()

    if model_lower.startswith("openrouter/"):
        return "openrouter"
    elif model_lower.startswith("xai/") or any(
        name in model_lower for name in ["grok"]
    ):
        return "xai"
    elif any(name in model_lower for name in ["gpt", "openai", "o1-"]):
        return "openai"
    elif any(name in model_lower for name in ["claude", "anthropic"]):
        return "anthropic"
    elif "gemini" in model_lower:
        return "gemini"
    elif "ollama" in model_lower:
        return "ollama"
    elif "deepseek" in model_lower:
        return "deepseek"
    else:
        return "unknown"


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
    is_being_piped: bool = False,
    raw_output: bool = False,
):
    """Stream responses from the LLM and format them using Rich."""
    try:
        # Add images if provided
        if images:
            # For models that expect base64
            if any(
                name in model.lower()
                for name in ["gpt-4", "gemini", "claude-3", "deepseek", "o1-", "grok"]
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
                # Update the content of the first message instead of replacing messages
                messages[0]["content"] = [
                    {"type": "text", "text": prompt},
                    *image_contents,
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
                        # Update the content of the first message
                        messages[0]["content"] = [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
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

        # Check if this is a reasoning model and reasoning is requested
        supports_reasoning = is_reasoning_model(model)
        provider = get_model_provider(model)

        # Use our flicker-free streaming renderer for non-piped output (unless raw mode)
        if not is_being_piped and not raw_output:
            renderer = StreamingMarkdownRenderer()

            # Create a loading indicator and connect it to the renderer
            from .streaming_markdown import LoadingIndicator

            loading_indicator = LoadingIndicator(sys.stdout)
            renderer.set_loading_indicator(loading_indicator)

            # Initialize timing for first loading indicator show
            last_chunk_time = time.time()
            first_content_received = False

            try:
                # For reasoning models with reasoning enabled
                if supports_reasoning and show_reasoning:
                    # For direct DeepSeek API (non-OpenRouter)
                    if provider == "deepseek" and not model.lower().startswith(
                        "openrouter/"
                    ):
                        client = OpenAI(
                            api_key=os.getenv("DEEPSEEK_API_KEY"),
                            base_url="https://api.deepseek.com",
                        )

                        # Convert messages to proper type
                        typed_messages_live: List[ChatCompletionMessageParam] = []
                        for msg in messages:
                            typed_messages_live.append(msg)  # type: ignore

                        response_stream = client.chat.completions.create(
                            model=model.split("/")[-1],  # Remove 'deepseek/' prefix
                            messages=typed_messages_live,
                            stream=True,
                        )

                        # Start loading indicator initially
                        loading_indicator.start("initial")

                        for chunk in response_stream:
                            if (
                                hasattr(chunk.choices[0].delta, "reasoning_content")
                                and chunk.choices[0].delta.reasoning_content  # type: ignore
                            ):
                                reasoning = chunk.choices[0].delta.reasoning_content  # type: ignore
                                accumulated_reasoning += reasoning
                                # For reasoning content, we could show it differently
                                # but for now, we'll focus on the main content
                            elif chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                # Transition from reasoning to content phase
                                if in_reasoning_phase and accumulated_reasoning:
                                    in_reasoning_phase = False

                                accumulated_content += content

                                if raw_output:
                                    # Raw output: write content directly
                                    sys.stdout.write(content)
                                    sys.stdout.flush()
                                else:
                                    renderer.add_text(content)

                                    # Stop initial loading indicator once content starts
                                    if not first_content_received:
                                        loading_indicator.stop()
                                        first_content_received = True
                    else:
                        # Use litellm for OpenRouter and other reasoning models
                        response_stream = completion(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=True,
                        )

                        # Start loading indicator
                        loading_indicator.start("initial")

                        for chunk in response_stream:
                            has_content = False

                            # Extract content from the chunk
                            delta = chunk.choices[0].delta  # type: ignore
                            content = delta.get("content", "")

                            # Check for reasoning content (may vary by provider)
                            reasoning_content = delta.get("reasoning_content", "")

                            if reasoning_content:
                                accumulated_reasoning += reasoning_content
                                has_content = True
                                # For reasoning content, we could show it differently
                                # but for now, we'll focus on the main content
                            elif content:
                                # Transition from reasoning to content phase
                                if in_reasoning_phase and accumulated_reasoning:
                                    in_reasoning_phase = False

                                accumulated_content += content

                                if raw_output:
                                    # Raw output: write content directly
                                    sys.stdout.write(content)
                                    sys.stdout.flush()
                                else:
                                    renderer.add_text(content)
                                has_content = True

                            # Only stop loading indicator and update time when we get actual content
                            if has_content and not raw_output:
                                if not first_content_received:
                                    loading_indicator.stop()
                                    first_content_received = True
                else:
                    # Use litellm for all non-reasoning models or when reasoning is disabled
                    response_stream = completion(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )

                    # Start loading indicator
                    loading_indicator.start("initial")

                    for chunk in response_stream:
                        # Extract content from the chunk
                        delta = chunk.choices[0].delta  # type: ignore
                        content = delta.get("content", "")

                        if content:
                            # Stop loading indicator once content starts
                            if not first_content_received and not raw_output:
                                loading_indicator.stop()
                                first_content_received = True

                            accumulated_content += content

                            if raw_output:
                                # Raw output: write content directly
                                sys.stdout.write(content)
                                sys.stdout.flush()
                            else:
                                renderer.add_text(content)

            except KeyboardInterrupt:
                if not raw_output:
                    loading_indicator.stop()
                    console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
                else:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
            except Exception as e:
                if not raw_output:
                    loading_indicator.stop()
                    console.print(f"[red]❌ Error: {e}[/red]")
                else:
                    sys.stderr.write(f"Error: {e}\n")
                    sys.stderr.flush()
                raise
            finally:
                # Always finalize the renderer and stop loading indicator (if not raw mode)
                if not raw_output:
                    loading_indicator.stop()
                    renderer.finalize()
                elif accumulated_content and not accumulated_content.endswith("\n"):
                    # Add final newline for raw output
                    sys.stdout.write("\n")
                    sys.stdout.flush()

        elif is_being_piped or raw_output:
            # Direct streaming for piped output or raw output (plain text, no markdown rendering)
            if supports_reasoning and show_reasoning:
                # For direct DeepSeek API (non-OpenRouter)
                if provider == "deepseek" and not model.lower().startswith(
                    "openrouter/"
                ):
                    client = OpenAI(
                        api_key=os.getenv("DEEPSEEK_API_KEY"),
                        base_url="https://api.deepseek.com",
                    )

                    # Convert messages to proper type
                    typed_messages: List[ChatCompletionMessageParam] = []
                    for msg in messages:
                        typed_messages.append(msg)  # type: ignore

                    response_stream = client.chat.completions.create(
                        model=model.split("/")[-1],
                        messages=typed_messages,
                        stream=True,
                    )

                    for chunk in response_stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            # Write content directly, preserving line breaks
                            sys.stdout.write(content.replace("\\n", "\n"))
                            sys.stdout.flush()
                            accumulated_content += content
                else:
                    # Use litellm for OpenRouter and other reasoning models
                    response_stream = completion(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )

                    for chunk in response_stream:
                        delta = chunk.choices[0].delta  # type: ignore
                        content = delta.get("content", "")
                        if content:
                            # Write content directly, preserving line breaks
                            sys.stdout.write(content.replace("\\n", "\n"))
                            sys.stdout.flush()
                            accumulated_content += content
            else:
                response_stream = completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )

                for chunk in response_stream:
                    delta = chunk.choices[0].delta  # type: ignore
                    content = delta.get("content", "")
                    if content:
                        # Write content directly, preserving line breaks
                        sys.stdout.write(content.replace("\\n", "\n"))
                        sys.stdout.flush()
                        accumulated_content += content

            # Add final newline for piped output
            if not accumulated_content.endswith("\n") and is_being_piped:
                sys.stdout.write("\n")
                sys.stdout.flush()

    except Exception as e:
        if not is_being_piped:
            console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@app.command(context_settings={"ignore_unknown_options": True})
def chat(
    prompt: list[str] = typer.Argument(..., help="The prompt to send to the LLM"),
    model: str = typer.Option(
        "gemini/gemini-2.5-flash",
        "--model",
        "-m",
        help="The LLM model to use. Examples: gpt-4o, claude-3-sonnet-20240229, ollama/llama2",
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
        help="Show the model's reasoning process (works with reasoning models like DeepSeek, OpenAI o1, etc.)",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version information and exit",
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        "--md",
        help="Output raw markdown without Rich formatting",
    ),
):
    """Chat with an LLM model and get markdown-formatted responses. Supports image input for compatible models."""

    # Check if we're being piped to another command
    is_being_piped = not sys.stdout.isatty()

    # Only show debug info if we're not being piped
    if not is_being_piped:
        print("Starting chat function...")  # Debug print

        if debug:
            print("Debug mode enabled")  # Basic print for debugging
            # litellm.set_verbose = True  # Not available in current version

    # Join the prompt list into a single string
    prompt_text = " ".join(prompt)
    display_text = prompt_text

    # Prepare the message content
    message_content = prompt_text

    # Check for piped input
    if not sys.stdin.isatty():
        piped_input = sys.stdin.read().strip()
        if piped_input:
            # Format the message with prompt first, then previous output
            message_content = f"{prompt_text}\n\n{piped_input}"
            display_text = f"{prompt_text}\n\n<Previous output>\n"

    # If context file is provided, read it and append to both display and message
    if context:
        try:
            with open(context, "r") as f:
                context_content = f.read()
                display_text = f"{display_text}\n\n# {os.path.basename(context)}\n..."
                message_content = f"{message_content}\n\nHere's the content of {os.path.basename(context)}:\n\n{context_content}"
        except Exception as e:
            console.print(f"[red]Error reading context file: {str(e)}[/red]")
            sys.exit(1)

    # Create the messages list
    messages = [{"role": "user", "content": message_content}]

    # Only show prompt info if we're not being piped
    if not is_being_piped:
        print(f"Prompt: {display_text}")  # Debug print

    # Validate and check API keys based on the model
    provider = get_model_provider(model)

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            console.print(
                "[red]Error: OPENROUTER_API_KEY environment variable is not set[/red]"
            )
            sys.exit(1)
        if debug and not is_being_piped:
            console.print(
                f"[dim]Found OpenRouter API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
            )
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print(
                "[red]Error: OPENAI_API_KEY environment variable is not set[/red]"
            )
            sys.exit(1)
        if debug and not is_being_piped:
            console.print(
                f"[dim]Found OpenAI API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
            )
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            console.print(
                "[red]Error: ANTHROPIC_API_KEY environment variable is not set[/red]"
            )
            sys.exit(1)
        if debug and not is_being_piped:
            console.print(
                f"[dim]Found Anthropic API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
            )
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            console.print(
                "[red]Error: GEMINI_API_KEY environment variable is not set[/red]"
            )
            sys.exit(1)
        if debug and not is_being_piped:
            console.print(
                f"[dim]Found Gemini API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
            )
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            console.print(
                "[red]Error: DEEPSEEK_API_KEY environment variable is not set[/red]"
            )
            sys.exit(1)
        if debug and not is_being_piped:
            console.print(
                f"[dim]Found DeepSeek API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
            )
    elif provider == "xai":
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            console.print(
                "[red]Error: XAI_API_KEY environment variable is not set[/red]"
            )
            sys.exit(1)
        if debug and not is_being_piped:
            console.print(
                f"[dim]Found xAI API key: {api_key[:4]}...{api_key[-4:]}[/dim]"
            )
    elif provider == "ollama":
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

    # Show what model we're using (only if not being piped)
    if not is_being_piped:
        console.print(f"[dim]Using model: {model}[/dim]")
        if images:
            console.print(
                f"[dim]With {len(images)} image{'s' if len(images) > 1 else ''}[/dim]"
            )
        console.print()  # Add a blank line for cleaner output

    # Configure model-specific settings
    if "ollama" in model.lower():
        # litellm.set_verbose = False  # Not available in current version
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
            is_being_piped=is_being_piped,  # Pass pipe status to response handler
            raw_output=raw,
        )
    except Exception as e:
        if not is_being_piped:
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
