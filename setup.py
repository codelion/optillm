import os
from setuptools import setup, find_packages

setup(
    name="optillm",
    version="0.1.18",
    packages=find_packages(include=['optillm', 'optillm.*']),  # This ensures all subpackages are included
    py_modules=['optillm'],
    package_data={
        'optillm': [
            'plugins/*.py',  # Include plugin files
            'cepo/*.py',     # Include cepo module Python files
            'cepo/configs/*.yaml',  # Include yaml files
        ],
    },
    include_package_data=True,  # This is important
    install_requires=[
        "numpy",
        "networkx",
        "openai",
        "z3-solver",
        "aiohttp",
        "flask",
        "torch",
        "transformers",
        "azure-identity",
        "tiktoken",
        "scikit-learn",
        "litellm",
        "requests",
        "beautifulsoup4",
        "lxml",
        "presidio_analyzer",
        "presidio_anonymizer",
        "nbconvert",
        "nbformat",
        "ipython",
        "ipykernel",
        "peft",
        "bitsandbytes",
        "gradio<5.16.0",
        # Constrain spacy version to avoid blis build issues on ARM64
        "spacy<3.8.0",
        "cerebras_cloud_sdk",
        "outlines[transformers]",
        "sentencepiece",
        "mcp",
        "adaptive-classifier",
        # MLX support for Apple Silicon optimization
        'mlx-lm>=0.24.0; platform_machine=="arm64" and sys_platform=="darwin"',
    ],
    entry_points={
        'console_scripts': [
            'optillm=optillm:main',  # Points directly to the main function in optillm.py
        ],
    },
    author="codelion",
    author_email="codelion@okyasoft.com",
    description="An optimizing inference proxy for LLMs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/codelion/optillm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
