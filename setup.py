"""
Setup script for Home Assistant Realtime Voice Assistant
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
    name="ha-realtime-voice-assistant",
    version="0.2.0-beta",
    description="A standalone Raspberry Pi voice assistant for Home Assistant using OpenAI Realtime API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nicholas Tripp",
    author_email="",
    url="https://github.com/nicholastripp/ha-realtime-assist",
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
        ],
        "wake-word": [
            "openwakeword>=0.1.0"
        ],
        "porcupine": [
            "pvporcupine>=3.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "ha-voice-assistant=main:main",
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
    keywords="home-assistant voice-assistant openai raspberry-pi smart-home",
    project_urls={
        "Bug Reports": "https://github.com/nicholastripp/ha-realtime-assist/issues",
        "Source": "https://github.com/nicholastripp/ha-realtime-assist",
        "Documentation": "https://github.com/nicholastripp/ha-realtime-assist/wiki",
    },
)