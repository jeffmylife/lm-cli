"""
Flicker-free streaming markdown renderer for terminal output.

This module provides a streaming markdown renderer that avoids the flickering
and scrolling issues common with real-time markdown rendering approaches.
"""

from __future__ import annotations
import sys
import time
import re
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
    4. Code blocks getting split during streaming

    The solution uses a buffer-based approach that tracks markdown state
    and only renders complete sections.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        output: Optional[TextIO] = None,
        min_render_interval: float = 0.05,  # Reduced for more responsive rendering
        min_content_threshold: int = 20,  # Reduced threshold
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

        # State tracking for markdown elements
        self.in_code_block = False
        self.code_fence_count = 0
        self.pending_code_start = ""

    def add_text(self, text: str) -> None:
        """
        Add text to the buffer and potentially render.

        Args:
            text: Text to add to the streaming buffer
        """
        self.buffer += text
        self._update_markdown_state()
        self._maybe_render()

    def _update_markdown_state(self) -> None:
        """Update internal state based on buffer content."""
        # Count code fence markers in the entire buffer
        # This handles ``` and ~~~
        fence_pattern = r"^```|^~~~"
        lines = self.buffer.split("\n")

        fence_count = 0
        for line in lines:
            if re.match(fence_pattern, line.strip()):
                fence_count += 1

        # If we have an odd number of fences, we're inside a code block
        self.in_code_block = (fence_count % 2) == 1
        self.code_fence_count = fence_count

    def _maybe_render(self) -> None:
        """Render if conditions are met to avoid flickering."""
        current_time = time.time()

        # Always check if we should force render first
        should_force = self._should_force_render()

        # Only render if enough time has passed OR we should force render
        if (
            current_time - self.last_render_time < self.min_render_interval
            and not should_force
        ):
            return

        # Check if we have enough new content to justify a render
        new_content_length = len(self.buffer) - self.last_rendered_length
        if new_content_length < self.min_content_threshold and not should_force:
            return

        self._render_new_content()
        self.last_render_time = current_time

    def _should_force_render(self) -> bool:
        """Check if we should force a render due to content structure."""
        # Never render if we're in the middle of a code block
        if self.in_code_block:
            return False

        # Force render on structural boundaries
        buffer_end = self.buffer[-50:] if len(self.buffer) > 50 else self.buffer

        force_conditions = [
            # Paragraph breaks
            self.buffer.endswith("\n\n"),
            # Headers
            self.buffer.endswith("\n# "),
            self.buffer.endswith("\n## "),
            self.buffer.endswith("\n### "),
            self.buffer.endswith("\n#### "),
            self.buffer.endswith("\n##### "),
            self.buffer.endswith("\n###### "),
            # Lists
            self.buffer.endswith("\n- "),
            self.buffer.endswith("\n* "),
            self.buffer.endswith("\n+ "),
            re.search(r"\n\d+\. $", self.buffer),  # Numbered lists
            # Block quotes
            self.buffer.endswith("\n> "),
            # Code block end
            self.buffer.endswith("```\n") and not self.in_code_block,
            self.buffer.endswith("~~~\n") and not self.in_code_block,
            # Horizontal rules
            self.buffer.endswith("\n---\n"),
            self.buffer.endswith("\n***\n"),
            self.buffer.endswith("\n___\n"),
            # Table rows
            self.buffer.endswith("|\n"),
            # End of sentences in paragraphs
            re.search(r"[.!?]\s*\n$", self.buffer),
        ]

        return any(force_conditions)

    def _find_safe_render_point(self) -> int:
        """Find a safe point to render up to, avoiding breaking markdown elements."""
        if len(self.buffer) <= self.last_rendered_length:
            return self.last_rendered_length

        # If we're in a code block, don't render past the last complete line before it
        if self.in_code_block:
            # Find the start of the current code block
            lines = self.buffer.split("\n")
            code_start_line = -1
            fence_count = 0

            for i, line in enumerate(lines):
                if re.match(r"^```|^~~~", line.strip()):
                    fence_count += 1
                    if fence_count % 2 == 1:  # Opening fence
                        code_start_line = i
                        break

            if code_start_line > 0:
                # Render up to the line before the code block
                safe_point = len("\n".join(lines[:code_start_line])) + 1
                return max(safe_point, self.last_rendered_length)
            else:
                # Don't render anything new if we can't find the code block start
                return self.last_rendered_length

        # Look for safe break points (complete paragraphs, headers, etc.)
        content_to_check = self.buffer[self.last_rendered_length :]
        lines = content_to_check.split("\n")

        safe_point = self.last_rendered_length
        current_pos = self.last_rendered_length

        for i, line in enumerate(
            lines[:-1]
        ):  # Don't include the last potentially incomplete line
            line_length = len(line) + 1  # +1 for newline
            current_pos += line_length

            # Check if this line ends a complete element
            if (
                line.strip() == ""  # Empty line (paragraph break)
                or line.strip().startswith("#")  # Header
                or line.strip().startswith("-")  # List item
                or line.strip().startswith("*")  # List item
                or line.strip().startswith("+")  # List item
                or re.match(r"^\d+\.", line.strip())  # Numbered list
                or line.strip().startswith(">")  # Block quote
                or line.strip() in ["---", "***", "___"]  # Horizontal rule
                or line.strip().endswith("|")  # Table row
                or re.search(r"[.!?]\s*$", line)  # End of sentence
            ):
                safe_point = current_pos

        return safe_point

    def _render_new_content(self) -> None:
        """Render only the new content since last render."""
        safe_point = self._find_safe_render_point()

        if safe_point <= self.last_rendered_length:
            return

        # Get the content to render
        content_to_render = self.buffer[self.last_rendered_length : safe_point]

        if not content_to_render.strip():
            return

        try:
            # Use md2term to convert the content
            convert(content_to_render)
            self.last_rendered_length = safe_point
        except Exception as e:
            # Fallback to plain text if markdown parsing fails
            self.output.write(content_to_render)
            self.output.flush()
            self.last_rendered_length = safe_point

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
        self.in_code_block = False
        self.code_fence_count = 0
        self.pending_code_start = ""


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
