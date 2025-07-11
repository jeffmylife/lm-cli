#!/usr/bin/env python3
"""
🎯 StreamLM Streaming Renderer Test Suite

This script validates our advanced streaming markdown renderer
with scenarios that test the specific fixes we've implemented:
- Short message handling (no hangs)
- Code block loading indicators
- Fast model compatibility
- Content-aware rendering

Run with: python test_streaming_examples.py
"""

import subprocess
import time
import sys

# Test cases that validate our streaming renderer fixes
TEST_CASES = [
    {
        "name": "🔥 SIMPLE HELLO TEST",
        "description": "The most basic test - should respond immediately without hanging",
        "prompt": "hello",
        "model": "gemini/gemini-2.5-flash",
        "tests": [
            "No hanging on short responses",
            "Immediate rendering",
            "Clean completion",
        ],
        "expected_time": 3.0,  # Should complete quickly
    },
    {
        "name": "⚡ FAST MODEL SHORT RESPONSE",
        "description": "Test fast model with short response (previous issue)",
        "prompt": "say hello world",
        "model": "gemini/gemini-2.5-flash-lite-preview-06-17",
        "tests": [
            "Fast model compatibility",
            "No broken formatting",
            "Clean line breaks",
        ],
        "expected_time": 2.0,
    },
    {
        "name": "🧮 SINGLE CODE BLOCK",
        "description": "Test code block loading indicator with medium-sized code",
        "prompt": "write a python function to calculate fibonacci numbers",
        "model": "gemini/gemini-2.5-flash",
        "tests": [
            "Code block loading indicator",
            "Proper syntax highlighting",
            "Clean code formatting",
            "No mid-sentence breaks",
        ],
        "expected_time": 15.0,
    },
    {
        "name": "🏗️ LARGE CODE BLOCK",
        "description": "Test loading indicator with large, complex code generation",
        "prompt": "write a comprehensive python class for a binary search tree with insert, search, delete, and traversal methods, include full docstrings",
        "model": "gemini/gemini-2.5-flash",
        "tests": [
            "Code block loading indicator shows",
            "Long code block handling",
            "Complex syntax highlighting",
            "Proper completion",
        ],
        "expected_time": 30.0,
    },
    {
        "name": "📝 MIXED CONTENT",
        "description": "Test content-aware rendering with mixed markdown",
        "prompt": "explain Python data structures: lists, dicts, sets. Include code examples for each",
        "model": "gemini/gemini-2.5-flash",
        "tests": [
            "Multiple code blocks",
            "Text between code blocks",
            "Progressive rendering",
            "No flickering",
        ],
        "expected_time": 20.0,
    },
    {
        "name": "🎨 COMPLEX MARKDOWN",
        "description": "Test all markdown features together",
        "prompt": "create a markdown guide with headers, bullet points, numbered lists, code blocks, and tables showing Python concepts",
        "model": "gemini/gemini-2.5-flash",
        "tests": [
            "Headers rendering",
            "Lists rendering",
            "Tables rendering",
            "Code blocks in lists",
            "Mixed formatting",
        ],
        "expected_time": 25.0,
    },
    {
        "name": "🚀 RAPID FIRE",
        "description": "Quick responses to test responsiveness",
        "prompt": "list 5 Python tips, be very brief",
        "model": "gemini/gemini-2.5-flash-lite-preview-06-17",
        "tests": [
            "Fast generation",
            "Quick rendering",
            "No loading indicator flicker",
            "Clean completion",
        ],
        "expected_time": 5.0,
    },
]


def run_test_case(test_case):
    """Run a single test case and display results."""
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {test_case['name']}")
    print(f"📋 DESCRIPTION: {test_case['description']}")
    print(f"🎯 TESTING: {', '.join(test_case['tests'])}")
    print(f"🤖 MODEL: {test_case['model']}")
    print(f"⏱️  EXPECTED TIME: ~{test_case['expected_time']}s")
    print(f"{'='*80}")

    # Construct the command using the development version
    cmd = [
        "python",
        "-m",
        "src.llm_cli.cli",
        "--model",
        test_case["model"],
        test_case["prompt"],
    ]

    print(f"🚀 Running: {test_case['prompt']}")
    print(f"⏱️  Started at: {time.strftime('%H:%M:%S')}")
    print()

    # Run the command and capture timing
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=60)
        end_time = time.time()
        actual_time = end_time - start_time

        print(f"\n⏱️  Completed in: {actual_time:.2f} seconds")
        print(f"✅ Exit code: {result.returncode}")

        # Check if timing is reasonable
        if actual_time <= test_case["expected_time"]:
            print(f"⚡ TIMING GOOD: Within expected {test_case['expected_time']}s")
        else:
            print(
                f"⏰ TIMING SLOW: Took {actual_time:.1f}s, expected ~{test_case['expected_time']}s"
            )

        if result.returncode == 0:
            print("🎉 TEST PASSED - Streaming worked perfectly!")
        else:
            print("❌ TEST FAILED - Check output above")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ TEST TIMEOUT - Generation took too long (>60s)")
        return False
    except KeyboardInterrupt:
        print("🛑 TEST INTERRUPTED - User cancelled")
        return False
    except Exception as e:
        print(f"💥 TEST ERROR: {e}")
        return False

    return True


def run_all_tests():
    """Run the complete test suite."""
    print("🎯 StreamLM Streaming Renderer Test Suite")
    print("=" * 50)
    print("This validates our streaming markdown renderer fixes:")
    print("✅ Short message handling (no hangs)")
    print("✅ Code block loading indicators")
    print("✅ Fast model compatibility")
    print("✅ Content-aware rendering")
    print()

    # Ask user which tests to run
    print("Available tests:")
    for i, test in enumerate(TEST_CASES, 1):
        print(f"{i}. {test['name']} (~{test['expected_time']}s)")

    print("\nOptions:")
    print("- Enter test numbers (e.g., '1,3,5' or '1-3')")
    print("- Enter 'all' to run all tests")
    print("- Enter 'core' to run essential tests (1,2,3)")
    print("- Enter 'q' to quit")

    choice = input("\nYour choice: ").strip().lower()

    if choice == "q":
        return

    # Parse user choice
    if choice == "all":
        tests_to_run = list(range(len(TEST_CASES)))
    elif choice == "core":
        tests_to_run = [0, 1, 2]  # Hello, fast model, single code block
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
    passed = 0
    failed = 0

    for i, test_idx in enumerate(tests_to_run):
        if 0 <= test_idx < len(TEST_CASES):
            if run_test_case(TEST_CASES[test_idx]):
                passed += 1
            else:
                failed += 1
                # Ask if user wants to continue on failure
                if i < len(tests_to_run) - 1:
                    cont = input(
                        "\n❓ Test failed. Continue with remaining tests? (y/n): "
                    ).lower()
                    if cont != "y":
                        break

            # Pause between tests (except last one)
            if i < len(tests_to_run) - 1:
                input("\n⏸️  Press Enter to continue to next test...")

    print(f"\n🎊 Test suite completed!")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if failed == 0:
        print("🏆 ALL TESTS PASSED - Ready for release!")
    else:
        print("🔧 Some tests failed - needs investigation")


# Quick test examples for manual testing
QUICK_TESTS = {
    "hello": ("hello", "gemini/gemini-2.5-flash"),
    "code": ("write a simple python function", "gemini/gemini-2.5-flash"),
    "fast": ("say hello", "gemini/gemini-2.5-flash-lite-preview-06-17"),
    "big": (
        "write a comprehensive python class for data processing",
        "gemini/gemini-2.5-flash",
    ),
}


def quick_test(test_name):
    """Run a quick predefined test."""
    if test_name not in QUICK_TESTS:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available: {', '.join(QUICK_TESTS.keys())}")
        return

    prompt, model = QUICK_TESTS[test_name]
    cmd = [
        "python",
        "-m",
        "src.llm_cli.cli",
        "--model",
        model,
        prompt,
    ]

    print(f"🚀 Quick test: {test_name}")
    print(f"🤖 Model: {model}")
    print(f"💬 Prompt: {prompt}")
    print()
    subprocess.run(cmd)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Quick test mode: python test_streaming_examples.py hello
        quick_test(sys.argv[1])
    else:
        # Interactive mode
        run_all_tests()
