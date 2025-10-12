"""
Setup script for OpenVoice MCP - Hybrid Multi-Server Voice Assistant
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().splitlines()
    # Filter out comments and empty lines
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="openvoice-mcp",
    version="2.0.0",
    description="Hybrid multi-server voice assistant with native/client-side MCP support for OpenAI Realtime API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nicholas Tripp",
    author_email="",
    url="https://github.com/nicholastripp/openvoice-mcp",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "openvoice-mcp=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Home Automation",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    keywords="mcp model-context-protocol voice-assistant openai raspberry-pi smart-home hybrid-architecture",
    project_urls={
        "Bug Reports": "https://github.com/nicholastripp/openvoice-mcp/issues",
        "Source": "https://github.com/nicholastripp/openvoice-mcp",
        "Documentation": "https://github.com/nicholastripp/openvoice-mcp/blob/main/README.md",
        "Base Project": "https://github.com/nicholastripp/ha-realtime-assist",
    },
)