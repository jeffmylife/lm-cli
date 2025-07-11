# Deployment Guide for LM CLI

This guide explains how to deploy new versions of LM CLI to PyPI and update the Homebrew formula.

## Prerequisites

1. **PyPI Account**: You need a PyPI account and API token
2. **GitHub Repository**: Push access to the main repository
3. **Homebrew Tap**: A separate repository for the Homebrew tap (optional)

## Setup

### 1. PyPI Configuration

Configure PyPI credentials:

```bash
# Create ~/.pypirc file
cat > ~/.pypirc << EOF
[distutils]
index-servers = pypi testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_API_TOKEN_HERE
EOF
```

### 2. Install Build Tools

```bash
pip install build twine
```

### 3. GitHub Actions Setup

For automated deployments, set up the following GitHub secrets:

- `PYPI_API_TOKEN`: Your PyPI API token
- `TEST_PYPI_API_TOKEN`: Your Test PyPI API token (optional)

## Deployment Process

### Method 1: Using the Deployment Script (Recommended)

```bash
# Test deployment to Test PyPI
python scripts/deploy.py --version 0.1.1 --test-pypi

# Production deployment to PyPI
python scripts/deploy.py --version 0.1.1 --publish
```

### Method 2: Manual Deployment

1. **Update Version**:
   ```bash
   # Update version in pyproject.toml
   sed -i 's/version = "0.1.0"/version = "0.1.1"/' pyproject.toml
   ```

2. **Build Package**:
   ```bash
   # Clean previous builds
   rm -rf dist/ build/ *.egg-info/
   
   # Build the package
   python -m build
   ```

3. **Test Package**:
   ```bash
   # Check package
   python -m twine check dist/*
   
   # Test installation
   pip install -e .
   lm --version
   ```

4. **Upload to PyPI**:
   ```bash
   # Upload to Test PyPI first
   python -m twine upload --repository testpypi dist/*
   
   # If successful, upload to PyPI
   python -m twine upload dist/*
   ```

5. **Create Git Tag**:
   ```bash
   git tag -a v0.1.1 -m "Release v0.1.1"
   git push origin v0.1.1
   ```

### Method 3: GitHub Actions (Automated)

1. **Create a Release**:
   - Go to GitHub → Releases → Create new release
   - Tag: `v0.1.1`
   - Title: `Release v0.1.1`
   - Description: List of changes

2. **Automatic Deployment**:
   - GitHub Actions will automatically build and publish to PyPI
   - Tests will run first to ensure quality

## Homebrew Formula Update

### Option 1: Using GitHub Actions

1. Go to the repository's Actions tab
2. Run the "Update Homebrew Formula" workflow
3. Enter the version number (e.g., `0.1.1`)
4. The formula will be automatically updated

### Option 2: Manual Update

1. **Generate Formula**:
   ```bash
   # Install homebrew-pypi-poet in a fresh environment
   python -m venv temp_env
   source temp_env/bin/activate
   pip install lm-cli==0.1.1 homebrew-pypi-poet
   
   # Generate formula
   poet -f lm-cli > homebrew/Formula/lm-cli.rb
   
   # Clean up
   deactivate
   rm -rf temp_env
   ```

2. **Update Formula Metadata**:
   ```bash
   # Update description and test block
   python3 -c "
   import re
   content = open('homebrew/Formula/lm-cli.rb').read()
   content = re.sub(r'desc \".*?\"', 'desc \"A command-line interface for interacting with various Large Language Models\"', content)
   content = re.sub(r'license \".*?\"', 'license \"MIT\"', content)
   content = re.sub(
       r'test do.*?end',
       'test do\\n    assert_match \"lm-cli version\", shell_output(\"#{bin}/lm --version\")\\n    assert_match \"Usage:\", shell_output(\"#{bin}/lm --help\")\\n  end',
       content,
       flags=re.DOTALL
   )
   open('homebrew/Formula/lm-cli.rb', 'w').write(content)
   "
   ```

3. **Test Formula**:
   ```bash
   # Test the formula locally
   HOMEBREW_NO_INSTALL_FROM_API=1 brew install --build-from-source homebrew/Formula/lm-cli.rb
   
   # Test the command
   lm --version
   
   # Uninstall
   brew uninstall lm-cli
   ```

## Post-Deployment

1. **Verify Installation**:
   ```bash
   # Test PyPI installation
   pip install lm-cli==0.1.1
   lm --version
   
   # Test Homebrew installation
   brew install jeffmylife/lm-cli/lm-cli
   lm --version
   ```

2. **Update Documentation**:
   - Update README.md with new features
   - Update CHANGELOG.md with release notes
   - Update version badges if needed

3. **Announce Release**:
   - Create GitHub release with changelog
   - Update any relevant documentation
   - Share on social media if appropriate

## Troubleshooting

### Common Issues

1. **PyPI Upload Fails**:
   - Check API token permissions
   - Ensure version number is incremented
   - Verify package builds correctly

2. **Homebrew Formula Issues**:
   - Check SHA256 hashes match
   - Ensure all dependencies are listed
   - Test formula on fresh system

3. **Version Conflicts**:
   - Ensure version is updated in all files
   - Check git tags don't conflict
   - Verify PyPI version matches git tag

### Getting Help

- Check GitHub Issues for known problems
- Review GitHub Actions logs for automated deployments
- Test locally before deploying

## Checklist

Before each release:

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run tests locally
- [ ] Build and test package locally
- [ ] Deploy to Test PyPI first
- [ ] Deploy to production PyPI
- [ ] Create git tag and push
- [ ] Update Homebrew formula
- [ ] Test installations
- [ ] Create GitHub release
- [ ] Update documentation 