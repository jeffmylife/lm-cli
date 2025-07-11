"""
Flicker-free streaming markdown renderer for terminal output.

This module provides a streaming markdown renderer that avoids the flickering
and scrolling issues common with real-time markdown rendering approaches.
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
from .markdown_renderer import convert


class LoadingIndicator:
    """An enhanced loading indicator with contextual messages for long generation periods."""

    def __init__(self, output: TextIO):
        self.output = output
        self.active = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.symbols = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_symbol = 0
        self.message_index = 0
        self.cycles_on_current_message = 0
        self.cycles_per_message = 15  # Change message every ~1.5 seconds (15 * 100ms)

        # Different messages to cycle through
        self.messages = [
            "generating content...",
            "assembling response...",
            "processing large code block...",
            "formatting markdown...",
            "thinking deeply...",
            "crafting detailed response...",
            "organizing information...",
            "finalizing content...",
        ]

    def start(self, context: Optional[str] = None):
        """Start the loading indicator with optional context."""
        if not self.active:
            self.active = True
            self.stop_event.clear()

            # Set context-specific message if provided
            if context == "code_block":
                self.messages = [
                    "processing large code block...",
                    "formatting code syntax...",
                    "organizing code structure...",
                    "finalizing code display...",
                    "assembling code block...",
                    "preparing syntax highlighting...",
                ]
            elif context == "initial":
                self.messages = [
                    "connecting to AI model...",
                    "sending your request...",
                    "waiting for response...",
                    "initializing generation...",
                ]
            else:
                # Default messages
                self.messages = [
                    "generating content...",
                    "assembling response...",
                    "processing large content...",
                    "formatting markdown...",
                    "thinking deeply...",
                    "crafting detailed response...",
                    "organizing information...",
                    "finalizing content...",
                ]

            self.message_index = 0
            self.cycles_on_current_message = 0
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the loading indicator and clear it."""
        if self.active:
            self.active = False
            self.stop_event.set()
            if self.thread:
                self.thread.join(timeout=0.1)
            self._clear_indicator()

    def _animate(self):
        """Animate the loading indicator with cycling messages."""
        while not self.stop_event.wait(0.1):  # 100ms intervals
            if self.active:
                symbol = self.symbols[self.current_symbol]
                message = self.messages[self.message_index]

                # Write the indicator with carriage return (no newline)
                indicator_text = f"\r{symbol} {message}"
                self.output.write(indicator_text)
                self.output.flush()

                # Update symbol
                self.current_symbol = (self.current_symbol + 1) % len(self.symbols)

                # Update message cycling
                self.cycles_on_current_message += 1
                if self.cycles_on_current_message >= self.cycles_per_message:
                    self.cycles_on_current_message = 0
                    self.message_index = (self.message_index + 1) % len(self.messages)

    def _clear_indicator(self):
        """Clear the loading indicator from the terminal."""
        # Calculate the maximum length of any message to clear properly
        max_message_length = max(len(msg) for msg in self.messages)
        clear_length = max_message_length + 3  # +3 for symbol and spaces

        # Clear the line and return cursor to beginning
        self.output.write("\r" + " " * clear_length + "\r")
        self.output.flush()


class StreamingMarkdownRenderer:
    """
    A flicker-free markdown renderer for streaming content.

    This renderer uses a simple buffering strategy:
    1. Buffers content until natural completion points
    2. Renders complete markdown elements as units
    3. No backtracking or clearing - just progressive output
    4. Shows loading indicator during long waits
    """

    def __init__(self, output: Optional[TextIO] = None):
        """Initialize the streaming renderer."""
        self.output = output or sys.stdout
        self.buffer = ""
        self.rendered_buffer = ""  # Track what we've already rendered
        self.last_render_time = 0.0
        self.last_output_time = 0.0  # Track when we last actually output content
        self.chars_since_render = 0
        self.in_code_block = False
        self.code_fence_count = 0
        self.loading_indicator: Optional[LoadingIndicator] = None

        # Timing thresholds
        self.min_render_interval = 0.1  # 100ms minimum between renders
        self.chars_threshold = 80  # Render after 80 characters
        self.time_threshold = 0.25  # Force render after 250ms
        self.force_render_chars = 40  # Minimum chars for time-based render

    def set_loading_indicator(self, indicator: LoadingIndicator) -> None:
        """Set the loading indicator to manage based on output activity."""
        self.loading_indicator = indicator

    def add_text(self, text: str) -> None:
        """Add text to the buffer and potentially trigger a render."""
        if not text:
            return

        self.buffer += text
        self.chars_since_render += len(text)

        # Update code block state
        self._update_code_block_state(text)

        current_time = time.time()

        # Determine if we should render
        should_render = False

        # Don't render while inside code blocks unless they're complete
        if self.in_code_block:
            # Code block is still open, don't render yet
            # But show loading indicator if we haven't output in a while
            if self.loading_indicator and current_time - self.last_output_time > 1.0:
                self.loading_indicator.start("code_block")
        else:
            # Force render on paragraph breaks
            if "\n\n" in text:
                should_render = True

            # Render if we've accumulated enough characters and enough time has passed
            elif (
                self.chars_since_render >= self.chars_threshold
                and current_time - self.last_render_time >= self.min_render_interval
            ):
                should_render = True

            # Force render if too much time has passed
            elif (
                current_time - self.last_render_time >= self.time_threshold
                and self.chars_since_render >= self.force_render_chars
            ):
                should_render = True

            # Render if content looks complete
            elif (
                self.chars_since_render >= 50
                and current_time - self.last_render_time >= self.min_render_interval
                and self._has_complete_elements()
            ):
                should_render = True

        # Also render when code block completes
        if not self.in_code_block and "```" in text:
            should_render = True

        if should_render:
            self._render_new_content()

    def _update_code_block_state(self, text: str) -> None:
        """Update the code block state based on new text."""
        # Count code fences in the new text
        fence_matches = re.findall(r"```", text)
        self.code_fence_count += len(fence_matches)

        # We're in a code block if we have an odd number of fences
        self.in_code_block = (self.code_fence_count % 2) == 1

    def _has_complete_elements(self) -> bool:
        """Check if buffer contains complete markdown elements."""
        if not self.buffer.strip():
            return False

        # Check for complete elements at the end
        text = self.buffer.rstrip()

        # Complete sentences (but be more flexible with punctuation)
        if re.search(r"[.!?]\s*$", text):
            return True

        # Complete paragraphs
        if text.endswith("\n\n"):
            return True

        # Complete headers
        if re.search(r"\n#{1,6}\s+.*\n\s*$", text):
            return True

        # Complete list items (more flexible detection)
        if re.search(r"\n\s*[-*+]\s+.*[.!?:]?\s*\n", text):
            return True

        # Complete numbered list items
        if re.search(r"\n\s*\d+\.\s+.*[.!?:]?\s*\n", text):
            return True

        # Complete sentences with commas (for better punctuation handling)
        if re.search(r"[,;:]\s*\n", text):
            return True

        return False

    def _render_new_content(self) -> None:
        """Render only the new content since last render."""
        if not self.buffer.strip():
            return

        # Only render if we have new content
        if self.buffer == self.rendered_buffer:
            return

        current_time = time.time()

        try:
            # Find a good break point for rendering
            break_point = self._find_good_break_point()

            if break_point <= len(self.rendered_buffer):
                return

            # Get the content up to the break point
            content_to_render = self.buffer[:break_point]
            new_content = content_to_render[len(self.rendered_buffer) :]

            if not new_content.strip():
                return

            # Render the new content as markdown
            from io import StringIO
            import sys

            old_stdout = sys.stdout
            captured_output = StringIO()
            sys.stdout = captured_output

            try:
                convert(new_content)
                rendered = captured_output.getvalue()
                sys.stdout = old_stdout

                # Write the new rendered content
                self.output.write(rendered)
                self.output.flush()

                # Stop loading indicator since we just output content
                if self.loading_indicator:
                    self.loading_indicator.stop()
                self.last_output_time = current_time

            finally:
                sys.stdout = old_stdout

            # Update tracking
            self.rendered_buffer = content_to_render
            self.last_render_time = current_time
            self.chars_since_render = len(self.buffer) - break_point

        except Exception:
            # Fallback: output new content as plain text
            new_content = self.buffer[len(self.rendered_buffer) :]
            self.output.write(new_content)
            self.output.flush()

            # Stop loading indicator since we just output content
            if self.loading_indicator:
                self.loading_indicator.stop()
            self.last_output_time = current_time

            self.rendered_buffer = self.buffer
            self.last_render_time = current_time
            self.chars_since_render = 0

    def _find_good_break_point(self) -> int:
        """Find a good point to break the buffer for rendering."""
        start_pos = len(self.rendered_buffer)
        remaining = self.buffer[start_pos:]

        if not remaining:
            return start_pos

        # Look for natural break points in order of preference
        break_patterns = [
            (r"\n\n", 0),  # Paragraph breaks (highest priority)
            (r"\n\s*[-*+]\s+.*?\n", 0),  # Complete list items
            (r"\n\s*\d+\.\s+.*?\n", 0),  # Complete numbered items
            (r"[.!?]\s*\n", 0),  # Sentence endings
            (r"[,:;]\s*\n", 0),  # Punctuation with newlines
            (r"\n#{1,6}\s+.*?\n", 0),  # Headers
        ]

        best_break = start_pos

        for pattern, offset in break_patterns:
            matches = list(re.finditer(pattern, remaining))
            if matches:
                # Take the last match for this pattern
                last_match = matches[-1]
                break_point = start_pos + last_match.end() - offset
                if break_point > best_break:
                    best_break = break_point

        # If no good break found, use character-based fallback
        if best_break == start_pos and len(remaining) > 120:
            # Find last word boundary in a reasonable chunk
            chunk_size = min(len(remaining), 100)
            chunk = remaining[:chunk_size]

            # Look for word boundaries
            for pos in reversed(range(len(chunk))):
                if chunk[pos] in " \n\t":
                    best_break = start_pos + pos + 1
                    break
            else:
                best_break = start_pos + chunk_size

        return min(best_break, len(self.buffer))

    def finalize(self) -> None:
        """Render any remaining content and finalize output."""
        if len(self.buffer) > len(self.rendered_buffer):
            self._render_new_content()
        # Add final newline
        self.output.write("\n")
        self.output.flush()
