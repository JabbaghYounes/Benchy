#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="edge-ai-benchmark",
    version="0.1.0",
    description="Edge AI Benchmark Suite for evaluating AI inference on edge devices",
    author="Edge AI Benchmark Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "psutil>=5.9.0",
        "requests>=2.28.0",
        "pyyaml>=6.0",
        "numpy>=1.21.0",
        "ultralytics>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "edge-bench=benchmark.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
