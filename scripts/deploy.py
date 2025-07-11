#!/usr/bin/env python3
"""
Deployment script for streamlm.

This script helps with the release process by:
1. Building the package
2. Checking the package
3. Uploading to PyPI (if --publish is used)
4. Creating a git tag
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    # Use uv pip for pip commands if uv is available, otherwise use python -m pip
    if cmd.startswith("pip "):
        # Check if uv is available
        uv_check = subprocess.run(
            "which uv", shell=True, capture_output=True, text=True
        )
        if uv_check.returncode == 0:
            cmd = cmd.replace("pip ", "uv pip ")
        else:
            cmd = cmd.replace("pip ", "python -m pip ")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result


def main():
    parser = argparse.ArgumentParser(description="Deploy streamlm")
    parser.add_argument("--version", required=True, help="Version to deploy")
    parser.add_argument("--publish", action="store_true", help="Publish to PyPI")
    parser.add_argument("--test-pypi", action="store_true", help="Publish to Test PyPI")
    args = parser.parse_args()

    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    print(f"Working in: {project_root}")

    # Update version in pyproject.toml and __init__.py
    import re

    # Update pyproject.toml
    pyproject_path = project_root / "pyproject.toml"
    with open(pyproject_path, "r") as f:
        content = f.read()

    pattern = r'version = "[^"]*"'
    replacement = f'version = "{args.version}"'
    new_content = re.sub(pattern, replacement, content)

    if new_content == content:
        print(f"Warning: Version pattern not found in pyproject.toml")
    else:
        with open(pyproject_path, "w") as f:
            f.write(new_content)
        print(f"Updated pyproject.toml version to {args.version}")

    # Update __init__.py
    init_path = project_root / "src" / "llm_cli" / "__init__.py"
    if init_path.exists():
        with open(init_path, "r") as f:
            init_content = f.read()

        init_pattern = r'__version__ = "[^"]*"'
        init_replacement = f'__version__ = "{args.version}"'
        new_init_content = re.sub(init_pattern, init_replacement, init_content)

        if new_init_content != init_content:
            with open(init_path, "w") as f:
                f.write(new_init_content)
            print(f"Updated __init__.py version to {args.version}")
        else:
            print(f"Warning: Version pattern not found in __init__.py")
    else:
        print(f"Warning: __init__.py not found at {init_path}")

    # Clean previous builds
    run_command("rm -rf dist/ build/ *.egg-info/")

    # Build the package
    run_command("python -m build")

    # Check the package
    run_command("python -m twine check dist/*")

    # Install and test locally
    print("Testing local installation...")
    run_command("pip install -e .")
    result = run_command("lm --version", check=False)
    if result.returncode != 0:
        print(f"Warning: 'lm --version' failed: {result.stderr}")
        # Try alternative method
        result = run_command(
            'python -c "from llm_cli import __version__; print(__version__)"',
            check=False,
        )

    if args.version not in result.stdout:
        print(
            f"Warning: Version check failed. Expected {args.version}, got: {result.stdout}"
        )
    else:
        print(f"✅ Version check passed: {result.stdout.strip()}")

    if args.publish:
        print("Publishing to PyPI...")
        run_command("python -m twine upload dist/*")
        print("✅ Published to PyPI!")
    elif args.test_pypi:
        print("Publishing to Test PyPI...")
        run_command("python -m twine upload --repository testpypi dist/*")
        print("✅ Published to Test PyPI!")
    else:
        print("Skipping publication (use --publish or --test-pypi)")

    # Create git tag
    run_command(f"git tag -a v{args.version} -m 'Release v{args.version}'")
    print(f"Created git tag v{args.version}")

    print("\n🎉 Deployment complete!")
    print(f"Next steps:")
    print(f"1. Push the tag: git push origin v{args.version}")
    print(f"2. Create a GitHub release")
    print(f"3. Update Homebrew formula using the GitHub Actions workflow")


if __name__ == "__main__":
    main()
