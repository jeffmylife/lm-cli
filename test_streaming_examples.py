#!/usr/bin/env python3
"""
🎯 StreamLM Streaming Renderer Test Suite

This script demonstrates our advanced streaming markdown renderer
with various real-world scenarios that stress-test different aspects
of the streaming functionality.

Run with: python test_streaming_examples.py
"""

import subprocess
import time
import sys

# Test cases that showcase different streaming scenarios
TEST_CASES = [
    {
        "name": "🔥 COMPLEX CODE GENERATION",
        "description": "Large, multi-method class with comprehensive docstrings",
        "prompt": "write a detailed python class for a binary search tree with insert, search, delete methods, include comprehensive docstrings",
        "tests": [
            "Large code blocks",
            "Complex syntax",
            "Nested methods",
            "Docstrings",
        ],
    },
    {
        "name": "📝 MARKDOWN LISTS & FORMATTING",
        "description": "Mixed content with bullets, numbers, and formatting",
        "prompt": "give me a comprehensive guide to Python data structures with bullet points, numbered steps, and code examples",
        "tests": [
            "Bullet points",
            "Numbered lists",
            "Mixed formatting",
            "Code snippets",
        ],
    },
    {
        "name": "🎨 RICH MARKDOWN FEATURES",
        "description": "Headers, tables, quotes, and complex formatting",
        "prompt": "create a markdown tutorial with headers, tables, blockquotes, and code blocks showing different Python concepts",
        "tests": ["Headers", "Tables", "Blockquotes", "Code blocks", "Mixed content"],
    },
    {
        "name": "⚡ RAPID STREAMING",
        "description": "Quick responses to test streaming responsiveness",
        "prompt": "list 10 Python tips in bullet points, be concise",
        "tests": ["Quick generation", "List formatting", "Responsiveness"],
    },
    {
        "name": "🧮 MULTIPLE CODE BLOCKS",
        "description": "Several separate code examples with explanations",
        "prompt": "show me 3 different Python sorting algorithms with explanations between each code block",
        "tests": ["Multiple code blocks", "Interleaved text", "Code separation"],
    },
    {
        "name": "📊 PUNCTUATION & SYMBOLS",
        "description": "Heavy punctuation usage and special characters",
        "prompt": "explain Python's special methods (__init__, __str__, __repr__, etc.) with examples, use lots of punctuation: commas, semicolons, colons!",
        "tests": ["Punctuation handling", "Special characters", "Symbol rendering"],
    },
]


def run_test_case(test_case):
    """Run a single test case and display results."""
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {test_case['name']}")
    print(f"📋 DESCRIPTION: {test_case['description']}")
    print(f"🎯 TESTING: {', '.join(test_case['tests'])}")
    print(f"{'='*80}")

    # Construct the command
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src.llm_cli.cli",
        "--model",
        "gemini/gemini-2.5-flash-lite-preview-06-17",
        test_case["prompt"],
    ]

    print(f"🚀 Running: {test_case['prompt'][:60]}...")
    print(f"⏱️  Started at: {time.strftime('%H:%M:%S')}")
    print()

    # Run the command and capture timing
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=120)
        end_time = time.time()

        print(f"\n⏱️  Completed in: {end_time - start_time:.2f} seconds")
        print(f"✅ Exit code: {result.returncode}")

        if result.returncode == 0:
            print("🎉 TEST PASSED - Streaming worked perfectly!")
        else:
            print("❌ TEST FAILED - Check output above")

    except subprocess.TimeoutExpired:
        print("⏰ TEST TIMEOUT - Generation took too long")
    except KeyboardInterrupt:
        print("🛑 TEST INTERRUPTED - User cancelled")
        return False
    except Exception as e:
        print(f"💥 TEST ERROR: {e}")

    return True


def run_all_tests():
    """Run the complete test suite."""
    print("🎯 StreamLM Streaming Renderer Test Suite")
    print("=" * 50)
    print("This will test our streaming markdown renderer with various scenarios")
    print("that stress-test different aspects of the streaming functionality.")
    print()

    # Ask user which tests to run
    print("Available tests:")
    for i, test in enumerate(TEST_CASES, 1):
        print(f"{i}. {test['name']}")

    print("\nOptions:")
    print("- Enter test numbers (e.g., '1,3,5' or '1-3')")
    print("- Enter 'all' to run all tests")
    print("- Enter 'q' to quit")

    choice = input("\nYour choice: ").strip().lower()

    if choice == "q":
        return

    # Parse user choice
    if choice == "all":
        tests_to_run = list(range(len(TEST_CASES)))
    else:
        tests_to_run = []
        try:
            for part in choice.split(","):
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    tests_to_run.extend(range(start - 1, end))
                else:
                    tests_to_run.append(int(part) - 1)
        except ValueError:
            print("❌ Invalid input format")
            return

    # Run selected tests
    print(f"\n🚀 Running {len(tests_to_run)} test(s)...")

    for i, test_idx in enumerate(tests_to_run):
        if 0 <= test_idx < len(TEST_CASES):
            if not run_test_case(TEST_CASES[test_idx]):
                break  # User interrupted

            # Pause between tests (except last one)
            if i < len(tests_to_run) - 1:
                input("\n⏸️  Press Enter to continue to next test...")

    print("\n🎊 Test suite completed!")


# Quick test examples for manual testing
QUICK_TESTS = {
    "code": "write a simple python function to calculate fibonacci",
    "list": "give me 5 reasons why Python is great, use bullet points",
    "mixed": "explain Python classes with a code example and bullet points",
    "complex": "write a detailed python class for a linked list with comprehensive methods",
}


def quick_test(test_name):
    """Run a quick predefined test."""
    if test_name not in QUICK_TESTS:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available: {', '.join(QUICK_TESTS.keys())}")
        return

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src.llm_cli.cli",
        "--model",
        "gemini/gemini-2.5-flash-lite-preview-06-17",
        QUICK_TESTS[test_name],
    ]

    print(f"🚀 Quick test: {test_name}")
    subprocess.run(cmd)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Quick test mode: python test_streaming_examples.py code
        quick_test(sys.argv[1])
    else:
        # Interactive mode
        run_all_tests()
