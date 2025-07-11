class LmCli < Formula
  include Language::Python::Virtualenv

  desc "A command-line interface for interacting with various Large Language Models"
  homepage "https://github.com/jeffmylife/streamlm"
  url "https://files.pythonhosted.org/packages/source/s/streamlm/streamlm-0.1.1.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"

  depends_on "python@3.12"

  # Dependencies will be auto-generated using homebrew-pypi-poet
  resource "litellm" do
    url "https://files.pythonhosted.org/packages/source/l/litellm/litellm-1.30.3.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/source/r/rich/rich-13.7.0.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "typer" do
    url "https://files.pythonhosted.org/packages/source/t/typer/typer-0.9.0.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "requests" do
    url "https://files.pythonhosted.org/packages/source/r/requests/requests-2.31.0.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "openai" do
    url "https://files.pythonhosted.org/packages/source/o/openai/openai-1.0.0.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "md2term" do
    url "https://files.pythonhosted.org/packages/source/m/md2term/md2term-1.0.0.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "streamlm version", shell_output("#{bin}/lm --version")
    assert_match "Usage:", shell_output("#{bin}/lm --help")
  end
end 