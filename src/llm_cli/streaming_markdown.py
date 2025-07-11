"""
Flicker-free streaming markdown renderer for terminal output.

This module provides a streaming markdown renderer that uses content-aware
rendering instead of timing-based thresholds for reliable, bug-free operation.
"""

from __future__ import annotations
import sys
import time
import re
import os
import threading
from typing import TextIO, Optional
from rich.console import Console
from rich.text import Text
import mistune
from .markdown_renderer.terminal_renderer import TerminalRenderer


class LoadingIndicator:
    """A simple loading indicator for initial connection phase."""

    def __init__(self, output: TextIO):
        self.output = output
        self.active = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.symbols = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_symbol = 0

    def start(self, context: Optional[str] = None):
        """Start the loading indicator."""
        if not self.active:
            self.active = True
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the loading indicator and clear it."""
        if self.active:
            self.active = False
            self.stop_event.set()
            if self.thread:
                self.thread.join(timeout=0.01)
            self._clear_indicator()

    def _animate(self):
        """Animate the loading indicator."""
        while not self.stop_event.wait(0.1):
            if self.active:
                symbol = self.symbols[self.current_symbol]
                self.output.write(f"\r{symbol} connecting...")
                self.output.flush()
                self.current_symbol = (self.current_symbol + 1) % len(self.symbols)

    def _clear_indicator(self):
        """Clear the loading indicator from the terminal."""
        self.output.write("\r" + " " * 20 + "\r")
        self.output.flush()


class StreamingMarkdownRenderer:
    """
    Content-aware streaming markdown renderer.

    Uses semantic boundaries in markdown content to determine safe render points,
    eliminating timing-based logic and ensuring reliable operation across all model speeds.
    """

    def __init__(self, output: Optional[TextIO] = None):
        """Initialize the streaming renderer."""
        self.output = output or sys.stdout
        self.console = Console(file=self.output, force_terminal=True)
        self.buffer = ""
        self.rendered_buffer = ""
        self.loading_indicator: Optional[LoadingIndicator] = None
        self.first_content_received = False

    def set_loading_indicator(self, indicator: LoadingIndicator) -> None:
        """Set the loading indicator to manage."""
        self.loading_indicator = indicator

    def add_text(self, text: str) -> None:
        """Add text to the buffer and render at safe points."""
        if not text:
            return

        # Stop loading indicator on first content
        if not self.first_content_received and self.loading_indicator:
            self.loading_indicator.stop()
            self.first_content_received = True

        # Add to buffer
        self.buffer += text

        # Find the latest safe render point
        safe_point = self._find_latest_safe_point()

        # Render if we found a safe point beyond what we've already rendered
        if safe_point > len(self.rendered_buffer):
            self._render_to_point(safe_point)

    def _find_latest_safe_point(self) -> int:
        """Find the furthest safe point in the buffer where we can render complete markdown structures."""
        # Only look at content we haven't rendered yet
        unrendered_start = len(self.rendered_buffer)
        unrendered_content = self.buffer[unrendered_start:]

        if not unrendered_content:
            return unrendered_start

        # Check if we're currently inside a code block by looking at ALL content (rendered + unrendered)
        # This is crucial - we need to count fences in the entire buffer to know our state
        total_fences_before_unrendered = self.rendered_buffer.count("```")
        inside_code_block = total_fences_before_unrendered % 2 == 1

        safe_points = []

        # If we're inside a code block, ONLY render when we find the closing fence
        if inside_code_block:
            closing_fence_match = re.search(r"```", unrendered_content)
            if closing_fence_match:
                # Include everything up to and including the closing fence
                safe_points.append(unrendered_start + closing_fence_match.end())
            # If no closing fence yet, don't render anything - wait for more content
            return max(safe_points) if safe_points else unrendered_start

        # Not inside a code block - look for complete markdown structures

        # 1. Complete code blocks (opening and closing fence in unrendered content)
        code_block_pattern = r"```[^\n]*\n.*?\n```"
        for match in re.finditer(code_block_pattern, unrendered_content, re.DOTALL):
            safe_points.append(unrendered_start + match.end())

        # 2. Complete sections ending with double newlines - but only if no open code blocks
        # We need to be extra careful here to not break ongoing code blocks
        for match in re.finditer(r"\n\n", unrendered_content):
            content_to_point = unrendered_content[: match.end()]
            # Check if this content has balanced code fences
            if content_to_point.count("```") % 2 == 0:
                safe_points.append(unrendered_start + match.end())

        # 3. Complete headers followed by content
        for match in re.finditer(
            r"(?:^|\n)(#{1,6})\s+[^\n]+\n", unrendered_content, re.MULTILINE
        ):
            # Only add if we're not in the middle of a code block
            content_to_point = unrendered_content[: match.end()]
            if content_to_point.count("```") % 2 == 0:
                safe_points.append(unrendered_start + match.end())

        # 4. For short responses, allow sentence boundaries
        if not safe_points and len(unrendered_content) < 150:
            # Check if it's just simple text without complex markdown
            if not re.search(r"```|(?:^|\n)#{1,6}\s", unrendered_content):
                for match in re.finditer(r"[.!?](?:\s+|$)", unrendered_content):
                    safe_points.append(unrendered_start + match.end())

        # Return the furthest safe point
        if safe_points:
            return max(safe_points)

        return unrendered_start

    def _has_balanced_code_fences(self, content: str) -> bool:
        """Check if content has balanced code fences (even number of ```)."""
        return content.count("```") % 2 == 0

    def _contains_incomplete_structures(self, content: str) -> bool:
        """Check if content contains incomplete markdown structures that shouldn't be rendered yet."""
        # Check for incomplete code blocks
        fence_count = content.count("```")
        if fence_count % 2 == 1:
            return True

        # Check for incomplete list items with code blocks
        lines = content.split("\n")
        in_list_item = False
        list_item_code_fences = 0

        for line in lines:
            # Check if this line starts a list item
            if re.match(r"^\s*([-*+]|\d+\.)\s", line):
                in_list_item = True
                list_item_code_fences = 0
            elif in_list_item:
                # Count code fences within this list item context
                if "```" in line:
                    list_item_code_fences += line.count("```")
                # If we hit a blank line or new list item, check if code blocks were complete
                if (
                    line.strip() == ""
                    or re.match(r"^\s*([-*+]|\d+\.)\s", line)
                    or not line.startswith(" ")
                    and not line.startswith("\t")
                ):
                    if list_item_code_fences % 2 == 1:
                        return True
                    in_list_item = False
                    list_item_code_fences = 0

        # Check if we ended while still in a list item with incomplete code blocks
        if in_list_item and list_item_code_fences % 2 == 1:
            return True

        return False

    def _contains_markdown_structures(self, content: str) -> bool:
        """Check if content contains any markdown structures."""
        # Check for headers, lists, code blocks, etc.
        if (
            "```" in content
            or re.search(r"(?:^|\n)#{1,6}\s", content)
            or re.search(r"(?:^|\n)\s*[-*+]\s", content)
            or re.search(r"(?:^|\n)\s*\d+\.\s", content)
        ):
            return True
        return False

    def _render_to_point(self, point: int) -> None:
        """Render content from current rendered position to the specified point."""
        if point <= len(self.rendered_buffer):
            return

        # Get the new content to render
        new_content = self.buffer[len(self.rendered_buffer) : point]

        if not new_content.strip():
            return

        # Use the original markdown renderer for proper formatting
        try:
            markdown = mistune.create_markdown(renderer=None)
            tokens = markdown(new_content)
            if isinstance(tokens, list):
                renderer = TerminalRenderer(self.console)
                renderer.render(tokens)
        except Exception:
            # Fallback: output as plain text if markdown rendering fails
            self.console.print(new_content, end="")

        # Update our rendered position
        self.rendered_buffer = self.buffer[:point]

    def finalize(self) -> None:
        """Render any remaining content and finalize output."""
        # Stop loading indicator if still active
        if self.loading_indicator:
            self.loading_indicator.stop()

        # Render any remaining content
        if len(self.buffer) > len(self.rendered_buffer):
            remaining_content = self.buffer[len(self.rendered_buffer) :]

            if remaining_content.strip():
                try:
                    # Use the original markdown renderer for proper formatting
                    markdown = mistune.create_markdown(renderer=None)
                    tokens = markdown(remaining_content)
                    if isinstance(tokens, list):
                        renderer = TerminalRenderer(self.console)
                        renderer.render(tokens)
                except Exception:
                    # Fallback to plain text
                    self.console.print(remaining_content, end="")

        # Add final newline
        self.console.print()
