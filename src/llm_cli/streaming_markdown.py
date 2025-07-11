"""
Flicker-free streaming markdown renderer for terminal output.

This module provides a streaming markdown renderer that avoids the flickering
and scrolling issues common with real-time markdown rendering approaches.
"""

from __future__ import annotations
import sys
import time
from typing import TextIO, Optional
from rich.console import Console
from md2term import convert


class StreamingMarkdownRenderer:
    """
    A flicker-free markdown renderer for streaming content.

    This renderer addresses the core issues with real-time markdown rendering:
    1. Flickering from constant re-rendering
    2. Scrolling interference from backtracking approaches
    3. Incomplete markdown structure causing parsing errors

    The solution uses a buffer-based approach that only renders complete
    markdown sections and never clears previous output.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        output: Optional[TextIO] = None,
        min_render_interval: float = 0.1,
        min_content_threshold: int = 50,
    ):
        """
        Initialize the streaming markdown renderer.

        Args:
            console: Rich console instance (optional, for compatibility)
            output: Output stream (defaults to sys.stdout)
            min_render_interval: Minimum time between renders in seconds
            min_content_threshold: Minimum characters before rendering
        """
        self.console = console
        self.output = output or sys.stdout
        self.buffer = ""
        self.last_rendered_length = 0
        self.last_render_time = 0.0
        self.min_render_interval = min_render_interval
        self.min_content_threshold = min_content_threshold

    def add_text(self, text: str) -> None:
        """
        Add text to the buffer and potentially render.

        Args:
            text: Text to add to the streaming buffer
        """
        self.buffer += text
        self._maybe_render()

    def _maybe_render(self) -> None:
        """Render if conditions are met to avoid flickering."""
        current_time = time.time()

        # Only render if enough time has passed and we have new content
        if (
            current_time - self.last_render_time < self.min_render_interval
            and not self._should_force_render()
        ):
            return

        # Check if we have enough new content to justify a render
        new_content_length = len(self.buffer) - self.last_rendered_length
        if (
            new_content_length < self.min_content_threshold
            and not self._should_force_render()
        ):
            return

        self._render_new_content()
        self.last_render_time = current_time

    def _should_force_render(self) -> bool:
        """Check if we should force a render due to content structure."""
        # Force render on paragraph breaks or structural elements
        return (
            self.buffer.endswith("\n\n")
            or self.buffer.endswith("\n# ")
            or self.buffer.endswith("\n## ")
            or self.buffer.endswith("\n### ")
            or self.buffer.endswith("\n#### ")
            or self.buffer.endswith("\n##### ")
            or self.buffer.endswith("\n###### ")
            or self.buffer.endswith("\n- ")
            or self.buffer.endswith("\n* ")
            or self.buffer.endswith("\n+ ")
            or self.buffer.endswith("\n> ")
            or self.buffer.endswith("\n```\n")
            or self.buffer.endswith("\n---\n")
            or self.buffer.endswith("\n***\n")
        )

    def _render_new_content(self) -> None:
        """Render only the new content since last render."""
        if len(self.buffer) <= self.last_rendered_length:
            return

        # Get the new content to render
        new_content = self.buffer[self.last_rendered_length :]

        # For terminal output, we'll render the new content as markdown
        # but only if it forms complete markdown elements
        if self._has_complete_elements(new_content):
            try:
                # Use md2term's convert function for the new content
                # This avoids the backtracking issues of StreamingRenderer
                convert(new_content)
                self.last_rendered_length = len(self.buffer)
            except Exception:
                # Fallback to plain text if markdown parsing fails
                self.output.write(new_content)
                self.output.flush()
                self.last_rendered_length = len(self.buffer)
        else:
            # For incomplete elements, just show plain text
            self.output.write(new_content)
            self.output.flush()
            self.last_rendered_length = len(self.buffer)

    def _has_complete_elements(self, content: str) -> bool:
        """Check if content contains complete markdown elements."""
        # Simple heuristic: render if we have complete lines or paragraphs
        return (
            "\n\n" in content
            or content.endswith("\n")
            or len(content) > 100  # Render long content even if incomplete
        )

    def finalize(self) -> None:
        """Render any remaining content and ensure proper termination."""
        if len(self.buffer) > self.last_rendered_length:
            remaining_content = self.buffer[self.last_rendered_length :]
            try:
                convert(remaining_content)
            except Exception:
                self.output.write(remaining_content)
                self.output.flush()

        # Add a final newline if needed
        if self.buffer and not self.buffer.endswith("\n"):
            self.output.write("\n")
            self.output.flush()

    def get_content(self) -> str:
        """Get the complete buffered content."""
        return self.buffer

    def clear(self) -> None:
        """Clear the buffer and reset state."""
        self.buffer = ""
        self.last_rendered_length = 0
        self.last_render_time = 0.0


def stream_markdown_to_terminal(
    content_generator,
    console: Optional[Console] = None,
    output: Optional[TextIO] = None,
) -> str:
    """
    Convenience function to stream markdown content to terminal.

    Args:
        content_generator: Iterator/generator that yields text chunks
        console: Rich console instance (optional)
        output: Output stream (defaults to sys.stdout)

    Returns:
        str: The complete rendered content
    """
    renderer = StreamingMarkdownRenderer(console=console, output=output)

    try:
        for chunk in content_generator:
            renderer.add_text(chunk)
    except KeyboardInterrupt:
        if console:
            console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        else:
            print("\n⚠️  Interrupted by user")
    finally:
        renderer.finalize()

    return renderer.get_content()


async def astream_markdown_to_terminal(
    content_generator,
    console: Optional[Console] = None,
    output: Optional[TextIO] = None,
) -> str:
    """
    Async version of stream_markdown_to_terminal.

    Args:
        content_generator: Async iterator/generator that yields text chunks
        console: Rich console instance (optional)
        output: Output stream (defaults to sys.stdout)

    Returns:
        str: The complete rendered content
    """
    renderer = StreamingMarkdownRenderer(console=console, output=output)

    try:
        async for chunk in content_generator:
            renderer.add_text(chunk)
    except KeyboardInterrupt:
        if console:
            console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        else:
            print("\n⚠️  Interrupted by user")
    finally:
        renderer.finalize()

    return renderer.get_content()
